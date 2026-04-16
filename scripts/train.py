"""QLoRA fine-tuning for all open-source models and tasks."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import click
import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).parent.parent
ALL_TASKS = ["ledgar", "cuad", "banking77", "fpb", "medmcqa", "mbpp"]
ALL_MODELS = ["qwen3-8b", "gemma3-4b", "phi4-mini"]
GPU_HOURLY = 1.19


class ModelConfig(BaseModel):
    model_id: str
    model_short: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: str = "bfloat16"
    fallback_model_id: Optional[str] = None
    lora: dict = Field(default_factory=dict)
    training: dict = Field(default_factory=dict)


class TaskConfig(BaseModel):
    task_id: str
    training_cap: Optional[int] = None
    efficiency_curve_sizes: list[int] = Field(default_factory=list)


def load_model_config(model_id: str) -> ModelConfig:
    path = REPO_ROOT / "configs" / "training" / f"{model_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return ModelConfig(**data)


def load_task_config(task_id: str) -> TaskConfig:
    path = REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return TaskConfig(**{k: v for k, v in data.items() if k in TaskConfig.model_fields})


def get_epochs(n_examples: int) -> int:
    if n_examples <= 200:
        return 10
    if n_examples <= 1000:
        return 5
    return 3


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def train_one(
    model_cfg: ModelConfig,
    task_id: str,
    condition: str,
    data_path: Path,
    dry_run: bool,
) -> dict:
    """Train a single model/task/condition combination. Returns metadata dict."""
    n_train = count_jsonl(data_path)
    adapter_dir = REPO_ROOT / "results" / "adapters" / model_cfg.model_short / task_id / condition
    log_dir = REPO_ROOT / "results" / "training" / model_cfg.model_short / task_id / condition
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    epochs = get_epochs(n_train)
    click.echo(f"  [{model_cfg.model_short}/{task_id}/{condition}] n={n_train}, epochs={epochs}")

    if dry_run:
        click.echo(f"  [dry-run] Would train {model_cfg.model_id} on {data_path.name}")
        meta = {"model_id": model_cfg.model_short, "task_id": task_id, "condition": condition,
                "n_train": n_train, "epochs": epochs, "training_cost": 0, "training_time_min": 0,
                "eval_loss": None, "model_used": model_cfg.model_id, "substituted": False}
        with open(log_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    # Lazy imports — only needed for actual training
    try:
        from unsloth import FastModel
        model_id = model_cfg.model_id
        substituted = False
    except ImportError:
        raise RuntimeError("Unsloth not installed. Run: pip install 'unsloth[cu124]'")

    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_id,
            max_seq_length=model_cfg.max_seq_length,
            load_in_4bit=model_cfg.load_in_4bit,
            dtype=None,  # auto
        )
    except Exception as exc:
        if model_cfg.fallback_model_id:
            click.echo(f"  WARNING: {model_id} failed ({exc}). Falling back to {model_cfg.fallback_model_id}")
            model_id = model_cfg.fallback_model_id
            substituted = True
            model, tokenizer = FastModel.from_pretrained(
                model_name=model_id,
                max_seq_length=model_cfg.max_seq_length,
                load_in_4bit=model_cfg.load_in_4bit,
                dtype=None,
            )
        else:
            raise

    lora_cfg = model_cfg.lora
    model = FastModel.get_peft_model(
        model,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_cfg.get("rank", 32),
        lora_alpha=lora_cfg.get("alpha", 64),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
        use_rslora=lora_cfg.get("use_rslora", True),
    )

    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    import datasets as hf_datasets

    # Load JSONL training data
    with open(data_path) as f:
        rows = [json.loads(line) for line in f]
    train_ds = hf_datasets.Dataset.from_list(rows)

    training_cfg = model_cfg.training
    sft_config = SFTConfig(
        output_dir=str(log_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=training_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.05),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        optim=training_cfg.get("optim", "adamw_8bit"),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        bf16=training_cfg.get("bf16", True),
        seed=training_cfg.get("seed", 42),
        save_strategy=training_cfg.get("save_strategy", "epoch"),
        eval_strategy=training_cfg.get("eval_strategy", "epoch"),
        load_best_model_at_end=training_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=training_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_cfg.get("greater_is_better", False),
        logging_steps=training_cfg.get("logging_steps", 10),
        report_to=training_cfg.get("report_to", "none"),
        max_seq_length=model_cfg.max_seq_length,
        dataset_text_field="messages",
        packing=False,
    )

    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_ds, args=sft_config)

    t0 = time.time()
    result = trainer.train()
    elapsed_min = (time.time() - t0) / 60
    training_cost = (elapsed_min / 60) * GPU_HOURLY

    # Save adapter
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    eval_loss = None
    if result.metrics:
        eval_loss = result.metrics.get("eval_loss")

    meta = {
        "model_id": model_cfg.model_short,
        "task_id": task_id,
        "condition": condition,
        "n_train": n_train,
        "epochs": epochs,
        "training_cost": round(training_cost, 4),
        "training_time_min": round(elapsed_min, 1),
        "eval_loss": eval_loss,
        "model_used": model_id,
        "substituted": substituted,
    }
    with open(log_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    click.echo(f"  Done: {elapsed_min:.1f} min, ${training_cost:.3f}, loss={eval_loss}")
    return meta


@click.command()
@click.option("--model", required=True, help="Model ID (qwen3-8b|gemma3-4b|phi4-mini)")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--condition", default="all", help="lora-500|lora-full|all")
@click.option("--efficiency-curve", is_flag=True, help="Run efficiency curve sizes for this model")
@click.option("--dry-run", is_flag=True, help="Validate configs without training")
def main(model: str, task: str, condition: str, efficiency_curve: bool, dry_run: bool) -> None:
    """QLoRA fine-tune open-source models on benchmark tasks."""
    model_ids = ALL_MODELS if model == "all" else [model]
    task_ids = ALL_TASKS if task == "all" else [task]
    conditions = ["lora-500", "lora-full"] if condition == "all" else [condition]

    failures = []
    for mid in model_ids:
        model_cfg = load_model_config(mid)
        for tid in task_ids:
            task_cfg = load_task_config(tid)
            prepared_dir = REPO_ROOT / "data" / "prepared" / tid

            if not efficiency_curve:
                for cond in conditions:
                    data_file = prepared_dir / ("train_500.jsonl" if cond == "lora-500" else "train_full.jsonl")
                    if not data_file.exists():
                        click.echo(f"  SKIP [{mid}/{tid}/{cond}]: {data_file} not found", err=True)
                        if not dry_run:
                            failures.append((f"{mid}/{tid}/{cond}", "data file missing"))
                        continue
                    try:
                        train_one(model_cfg, tid, cond, data_file, dry_run)
                    except Exception as exc:
                        click.echo(f"  ERROR [{mid}/{tid}/{cond}]: {exc}", err=True)
                        failures.append((f"{mid}/{tid}/{cond}", str(exc)))
            else:
                for n in task_cfg.efficiency_curve_sizes:
                    data_file = prepared_dir / f"train_{n}.jsonl"
                    if not data_file.exists():
                        click.echo(f"  SKIP [{mid}/{tid}/lora-{n}]: data not found")
                        continue
                    try:
                        train_one(model_cfg, tid, f"lora-{n}", data_file, dry_run)
                    except Exception as exc:
                        click.echo(f"  ERROR [{mid}/{tid}/lora-{n}]: {exc}", err=True)
                        failures.append((f"{mid}/{tid}/lora-{n}", str(exc)))

    if failures:
        click.echo(f"\nFAILED ({len(failures)}):")
        for key, err in failures:
            click.echo(f"  {key}: {err}")
        sys.exit(1)
    click.echo("\nAll training jobs completed.")


if __name__ == "__main__":
    main()
