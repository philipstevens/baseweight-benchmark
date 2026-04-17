"""QLoRA fine-tuning for Qwen3-8B on BANKING77 and CUAD."""
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
ALL_TASKS = ["banking77", "cuad"]
ALL_MODELS = ["qwen3-8b"]
GPU_HOURLY = 0.49  # RTX 4090 on RunPod ($0.44–0.54/hr midpoint)


class ModelConfig(BaseModel):
    model_id: str
    model_short: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: str = "bfloat16"
    enable_thinking: Optional[bool] = None
    fallback_model_id: Optional[str] = None
    lora: dict = Field(default_factory=dict)
    training: dict = Field(default_factory=dict)


class TaskConfig(BaseModel):
    task_id: str
    training_cap: Optional[int] = None
    max_seq_length: Optional[int] = None  # overrides model max_seq_length when set
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
    task_cfg: TaskConfig,
    condition: str,
    data_path: Path,
    dry_run: bool,
    auto_upload: bool = False,
) -> dict:
    """Train a single model/task/condition. Returns metadata dict."""
    task_id = task_cfg.task_id
    n_train = count_jsonl(data_path)
    seq_len = task_cfg.max_seq_length or model_cfg.max_seq_length

    adapter_dir = REPO_ROOT / "results" / "adapters" / model_cfg.model_short / task_id / condition
    log_dir = REPO_ROOT / "results" / "training" / model_cfg.model_short / task_id / condition
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    epochs = get_epochs(n_train)
    click.echo(f"  [{model_cfg.model_short}/{task_id}/{condition}] n={n_train}, epochs={epochs}, seq_len={seq_len}")

    if dry_run:
        click.echo(f"  [dry-run] Would train {model_cfg.model_id} on {data_path.name}")
        meta = {
            "model_id": model_cfg.model_short, "task_id": task_id, "condition": condition,
            "n_train": n_train, "epochs": epochs, "seq_len": seq_len,
            "training_cost": 0, "training_time_min": 0,
            "eval_loss": None, "model_used": model_cfg.model_id, "substituted": False,
        }
        with open(log_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    try:
        from unsloth import FastModel
        model_id = model_cfg.model_id
        substituted = False
    except ImportError:
        raise RuntimeError("Unsloth not installed. Run: pip install 'unsloth[cu124-torch260]'")

    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_id,
            max_seq_length=seq_len,
            load_in_4bit=model_cfg.load_in_4bit,
            dtype=None,
        )
    except Exception as exc:
        if model_cfg.fallback_model_id:
            click.echo(f"  WARNING: {model_id} failed ({exc}). Falling back to {model_cfg.fallback_model_id}")
            model_id = model_cfg.fallback_model_id
            substituted = True
            model, tokenizer = FastModel.from_pretrained(
                model_name=model_id,
                max_seq_length=seq_len,
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
    import datasets as hf_datasets

    with open(data_path) as f:
        rows = [json.loads(line) for line in f]
    train_ds = hf_datasets.Dataset.from_list(rows)

    # For Qwen3 with enable_thinking=False, pre-apply the chat template so
    # the <think> tokens are never generated during training.
    if model_cfg.enable_thinking is False:
        def apply_template(example):
            return {"text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )}
        train_ds = train_ds.map(apply_template)
        dataset_text_field = "text"
    else:
        dataset_text_field = "messages"

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
        max_seq_length=seq_len,
        dataset_text_field=dataset_text_field,
        packing=False,
    )

    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_ds, args=sft_config)

    t0 = time.time()
    result = trainer.train()
    elapsed_min = (time.time() - t0) / 60
    training_cost = (elapsed_min / 60) * GPU_HOURLY

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    eval_loss = result.metrics.get("eval_loss") if result.metrics else None

    meta = {
        "model_id": model_cfg.model_short,
        "task_id": task_id,
        "condition": condition,
        "n_train": n_train,
        "epochs": epochs,
        "seq_len": seq_len,
        "training_cost": round(training_cost, 4),
        "training_time_min": round(elapsed_min, 1),
        "eval_loss": eval_loss,
        "model_used": model_id,
        "substituted": substituted,
    }
    with open(log_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    click.echo(f"  Done: {elapsed_min:.1f} min, ${training_cost:.3f}, loss={eval_loss}")

    if auto_upload:
        _upload_adapter(model_cfg.model_short, task_id, condition)

    return meta


def _upload_adapter(model_short: str, task_id: str, condition: str) -> None:
    import subprocess
    click.echo(f"  Auto-uploading {model_short}/{task_id}/{condition} to HuggingFace...")
    result = subprocess.run(
        ["python", str(REPO_ROOT / "scripts" / "upload_artifacts.py"),
         "--model", model_short, "--task", task_id, "--condition", condition],
        capture_output=False,
    )
    if result.returncode != 0:
        click.echo(f"  WARNING: upload failed (exit {result.returncode})", err=True)


@click.command()
@click.option("--model", default="qwen3-8b", help="Model ID or 'all'")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--condition", default="all", help="lora-500|lora-full|all")
@click.option("--auto-upload", is_flag=True, help="Upload adapter to HuggingFace after each run (persistence)")
@click.option("--dry-run", is_flag=True, help="Validate configs without training")
def main(model: str, task: str, condition: str, auto_upload: bool, dry_run: bool) -> None:
    """QLoRA fine-tune Qwen3-8B on BANKING77 and CUAD."""
    model_ids = ALL_MODELS if model == "all" else [model]
    task_ids = ALL_TASKS if task == "all" else [task]
    conditions = ["lora-500", "lora-full"] if condition == "all" else [condition]

    failures = []
    for mid in model_ids:
        model_cfg = load_model_config(mid)
        for tid in task_ids:
            task_cfg = load_task_config(tid)  # load once per task, not per condition
            prepared_dir = REPO_ROOT / "data" / "prepared" / tid
            for cond in conditions:
                data_file = prepared_dir / ("train_500.jsonl" if cond == "lora-500" else "train_full.jsonl")
                if not data_file.exists():
                    click.echo(f"  SKIP [{mid}/{tid}/{cond}]: {data_file} not found", err=True)
                    if not dry_run:
                        failures.append((f"{mid}/{tid}/{cond}", "data file missing"))
                    continue
                try:
                    train_one(model_cfg, task_cfg, cond, data_file, dry_run, auto_upload=auto_upload)
                except Exception as exc:
                    click.echo(f"  ERROR [{mid}/{tid}/{cond}]: {exc}", err=True)
                    failures.append((f"{mid}/{tid}/{cond}", str(exc)))

    if failures:
        click.echo(f"\nFAILED ({len(failures)}):")
        for key, err in failures:
            click.echo(f"  {key}: {err}")
        sys.exit(1)
    click.echo("\nAll training jobs completed.")


if __name__ == "__main__":
    main()
