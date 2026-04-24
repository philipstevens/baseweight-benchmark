"""QLoRA fine-tuning for Qwen3-8B on BANKING77 and CUAD."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import click
import yaml
from pydantic import BaseModel, Field

from checkpoint_utils import (
    atomic_write_json,
    checkpoint_dir,
    find_hf_resume_checkpoint,
    load_train_state,
    save_train_state,
)

REPO_ROOT = Path(__file__).parent.parent
ALL_TASKS = ["banking77", "cuad"]
ALL_MODELS = ["qwen3-8b"]
GPU_HOURLY = 0.49  # Default GPU hourly rate — override via pricing.yaml


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
    tiny: bool = False,
) -> dict:
    """Train a single model/task/condition. Returns metadata dict."""
    task_id = task_cfg.task_id
    n_train = count_jsonl(data_path)
    seq_len = task_cfg.max_seq_length or model_cfg.max_seq_length

    adapter_dir = REPO_ROOT / "results" / "adapters" / model_cfg.model_short / task_id / condition
    log_dir = REPO_ROOT / "results" / "training" / model_cfg.model_short / task_id / condition
    ckpt_dir = checkpoint_dir(model_cfg.model_short, task_id, condition)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epochs = get_epochs(n_train)
    click.echo(f"  [{model_cfg.model_short}/{task_id}/{condition}] n={n_train}, epochs={epochs}, seq_len={seq_len}")

    prior_state = load_train_state(model_cfg.model_short, task_id, condition)
    if prior_state and prior_state.get("status") == "complete":
        click.echo(f"  SKIP [{model_cfg.model_short}/{task_id}/{condition}]: already complete")
        meta_path = log_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {}

    if dry_run:
        click.echo(f"  [dry-run] Would train {model_cfg.model_id} on {data_path.name}")
        meta = {
            "model_id": model_cfg.model_short, "task_id": task_id, "condition": condition,
            "n_train": n_train, "epochs": epochs, "seq_len": seq_len,
            "training_cost": 0, "training_time_min": 0,
            "eval_loss": None, "model_used": model_cfg.model_id, "substituted": False,
        }
        atomic_write_json(meta, log_dir / "metadata.json")
        return meta

    # Detect an existing checkpoint to resume from.
    resume_ckpt = find_hf_resume_checkpoint(model_cfg.model_short, task_id, condition)
    if resume_ckpt:
        click.echo(f"  Resuming from checkpoint: {resume_ckpt.name}")
    save_train_state(model_cfg.model_short, task_id, condition, {
        "status": "in_progress",
        "epoch": 0,
        "global_step": 0,
    })

    lora_cfg = model_cfg.lora
    model_id = model_cfg.model_id
    substituted = False

    # --tiny = Intel XPU backend (Unsloth installed with Intel GPU support)
    # no flag = CUDA backend (Unsloth installed with CUDA support)
    if tiny:
        os.environ.setdefault("UNSLOTH_DISABLE_AUTO_PADDING_FREE", "1")
        # Disable torch.dynamo/compile for XPU: our Python Tensor.to patch is not
        # traceable by dynamo (it wraps a C++ builtin), and Triton compilation fails
        # anyway (missing level_zero/ze_api.h). Eager mode is correct here.
        os.environ["TORCHDYNAMO_DISABLE"] = "1"

        # unsloth_zoo queries mem_get_info at import time; Intel iGPUs don't support it.
        # Patch before import so the call returns a plausible (free, total) tuple instead.
        import torch
        _orig_mem_get_info = torch.xpu.mem_get_info
        def _safe_mem_get_info(device=None):
            try:
                return _orig_mem_get_info(device)
            except RuntimeError:
                total = torch.xpu.get_device_properties(0).total_memory
                return (total, total)
        torch.xpu.mem_get_info = _safe_mem_get_info
        torch.xpu.memory.mem_get_info = _safe_mem_get_info

        # Intel Iris Xe iGPU (0x9a49) raises UR_RESULT_ERROR_INVALID_ARGUMENT (45) for
        # dtype conversions on XPU — float32 casts, bool casts, .float() calls, etc.
        #   - bool target: use comparison (tensor != 0) which XPU supports
        #   - other failures: round-trip via CPU (device transfers work; dtype casts don't)
        #   - float32/float16 redirected to bfloat16 (device's native float dtype)
        _orig_tensor_to = torch.Tensor.to

        def _xpu_safe_dtype_cast(self, target_dtype):
            if target_dtype == torch.bool:
                return (self != 0)
            effective = (
                torch.bfloat16 if target_dtype in (torch.float32, torch.float16)
                else target_dtype
            )
            try:
                cpu_t = _orig_tensor_to(self, torch.device("cpu"))
                if cpu_t.dtype != effective:
                    cpu_t = _orig_tensor_to(cpu_t, effective)
                return _orig_tensor_to(cpu_t, self.device)
            except Exception:
                return self

        def _xpu_persistent_tensor_to(self, *args, **kw):
            if self.device.type != "xpu":
                return _orig_tensor_to(self, *args, **kw)
            try:
                return _orig_tensor_to(self, *args, **kw)
            except RuntimeError as _e:
                if "UR error" not in str(_e):
                    raise
                _dtype = kw.get("dtype") or next(
                    (a for a in args if isinstance(a, torch.dtype)), None
                )
                if _dtype is not None:
                    return _xpu_safe_dtype_cast(self, _dtype)
                return self
        torch.Tensor.to = _xpu_persistent_tensor_to

        # .float() bypasses Tensor.to on the Python side
        _orig_tensor_float = torch.Tensor.float
        def _xpu_persistent_float(self):
            if self.device.type == "xpu":
                return _xpu_safe_dtype_cast(self, torch.float32)
            return _orig_tensor_float(self)
        torch.Tensor.float = _xpu_persistent_float

    try:
        from unsloth import FastModel
    except ImportError:
        backend_label = "Intel GPU" if tiny else "CUDA"
        raise RuntimeError(f"Unsloth not installed with {backend_label} support. See: https://unsloth.ai/docs/get-started/install")

    backend = "XPU" if tiny else "CUDA"
    click.echo(f"  Loading {model_id} on {backend}...")

    import torch as _torch
    # XPU supports bfloat16; load in bf16 so base model and LoRA adapters share the same dtype.
    # For CUDA the config dtype drives the choice (None = auto).
    _dtype = _torch.bfloat16 if tiny else None

    if tiny:
        # PEFT's cast_adapter_dtype calls param.data.to(torch.float32) on device tensors.
        # Disable autocast so it skips the cast entirely; our Tensor.to patch handles any
        # remaining calls, but avoiding them up-front is cleaner.
        import peft.tuners.tuners_utils as _peft_utils
        _orig_cast = _peft_utils.cast_adapter_dtype
        def _xpu_cast_adapter_dtype(model, adapter_name, autocast_adapter_dtype=True):
            _orig_cast(model, adapter_name, autocast_adapter_dtype=False)
        _peft_utils.cast_adapter_dtype = _xpu_cast_adapter_dtype

    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_id,
            max_seq_length=seq_len,
            load_in_4bit=model_cfg.load_in_4bit,
            dtype=_dtype,
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
                dtype=_dtype,
            )
        else:
            raise

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
        use_gradient_checkpointing=False if tiny else "unsloth",
    )

    if tiny:
        # XPU compute ops (pow, mean, etc.) fail on Intel Iris Xe Tiger Lake with
        # UR_RESULT_ERROR_INVALID_ARGUMENT. Device transfers work, compute doesn't.
        # Fall back to CPU for the actual training loop.
        click.echo("  Moving model to CPU (XPU compute not supported on this device)...")
        model = model.to("cpu")

    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback
    import datasets as hf_datasets

    class _CheckpointCallback(TrainerCallback):
        """Sync lightweight state to network volume after each checkpoint so training can resume on pod restart."""
        def on_save(self, args, state, control, **kwargs):
            save_train_state(model_cfg.model_short, task_id, condition, {
                "status": "in_progress",
                "epoch": state.epoch,
                "global_step": state.global_step,
                "best_metric": state.best_metric,
                "best_model_checkpoint": state.best_model_checkpoint,
            })

    with open(data_path) as f:
        rows = [json.loads(line) for line in f]
    train_ds = hf_datasets.Dataset.from_list(rows)

    # Pre-apply the chat template so trl tokenizes a plain "text" field.
    # When enable_thinking is explicitly False, suppress <think> tokens.
    template_kwargs = {}
    if model_cfg.enable_thinking is False:
        template_kwargs["enable_thinking"] = False

    def apply_template(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            **template_kwargs,
        )}
    train_ds = train_ds.map(apply_template)
    dataset_text_field = "text"

    training_cfg = model_cfg.training
    if tiny:
        sft_config = SFTConfig(
            output_dir=str(ckpt_dir),
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
            lr_scheduler_type="constant",
            warmup_ratio=0.0,
            weight_decay=training_cfg.get("weight_decay", 0.01),
            optim="adamw_torch",
            max_grad_norm=1.0,
            use_cpu=True,
            bf16=False,
            fp16=False,
            seed=training_cfg.get("seed", 42),
            gradient_checkpointing=False,
            save_strategy="no",
            eval_strategy="no",
            load_best_model_at_end=False,
            logging_steps=1,
            report_to="none",
            max_seq_length=min(seq_len, 256),
            dataset_text_field=dataset_text_field,
            packing=False,
        )
    else:
        sft_config = SFTConfig(
            output_dir=str(ckpt_dir),  # HF intermediate checkpoints → network volume
            num_train_epochs=epochs,
            per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
            learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
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

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=train_ds, args=sft_config,
        callbacks=[_CheckpointCallback()],
    )

    t0 = time.time()
    result = trainer.train(
        resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None
    )
    elapsed_min = (time.time() - t0) / 60
    training_cost = 0.0 if tiny else (elapsed_min / 60) * GPU_HOURLY

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Mirror to network volume so adapter survives pod termination
    # even if REPO_ROOT is on local pod disk.
    nv_adapter_dir = ckpt_dir / "final_adapter"
    shutil.copytree(str(adapter_dir), str(nv_adapter_dir), dirs_exist_ok=True)

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
    atomic_write_json(meta, log_dir / "metadata.json")
    atomic_write_json(meta, ckpt_dir / "metadata.json")
    save_train_state(model_cfg.model_short, task_id, condition, {
        "status": "complete",
        "eval_loss": eval_loss,
        "training_time_min": round(elapsed_min, 1),
        "training_cost": round(training_cost, 4),
    })

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
@click.option("--tiny", is_flag=True, help="Use XPU backend (Unsloth with Intel GPU support). Default uses CUDA.")
def main(model: str, task: str, condition: str, auto_upload: bool, dry_run: bool, tiny: bool) -> None:
    """QLoRA fine-tune on BANKING77 and CUAD via Unsloth (CUDA or Intel XPU)."""
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
                    train_one(model_cfg, task_cfg, cond, data_file, dry_run, auto_upload=auto_upload, tiny=tiny)
                except Exception as exc:
                    click.echo(f"  ERROR [{mid}/{tid}/{cond}]: {exc}", err=True)
                    traceback.print_exc()
                    failures.append((f"{mid}/{tid}/{cond}", str(exc)))

    if failures:
        click.echo(f"\nFAILED ({len(failures)}):")
        for key, err in failures:
            click.echo(f"  {key}: {err}")
        sys.exit(1)
    click.echo("\nAll training jobs completed.")


if __name__ == "__main__":
    main()
