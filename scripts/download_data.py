"""Download raw datasets from HuggingFace for all benchmark tasks."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import click
from pydantic import BaseModel, Field
import yaml

# Expected minimum row counts for sanity checks (split: min_count)
EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "banking77": {"train": 8000,  "test": 2000},
    "cuad":      {"train": 100},
    "ledgar":    {"train": 50000, "test": 5000},
    "fpb":       {"train": 3000},
    "medmcqa":   {"train": 100000, "test": 4000},
    "mbpp":      {"train": 374,   "test": 500},
}

REPO_ROOT = Path(__file__).parent.parent


class TaskConfig(BaseModel):
    task_id: str
    task_name: str
    dataset_path: str
    dataset_config: Optional[str] = None


def load_task_configs(task_ids: list[str]) -> list[TaskConfig]:
    configs = []
    config_dir = REPO_ROOT / "configs" / "tasks"
    for tid in task_ids:
        path = config_dir / f"{tid}.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        configs.append(TaskConfig(**{k: data[k] for k in TaskConfig.model_fields if k in data}))
    return configs


ALL_TASKS = ["banking77", "cuad", "ledgar", "fpb", "medmcqa", "mbpp"]


@click.command()
@click.option("--task", default=None, help="Task ID to download (required; use 'all' to download every task)")
@click.option("--dry-run", is_flag=True, help="Validate config without downloading")
def main(task: str, dry_run: bool) -> None:
    """Download benchmark datasets from HuggingFace.

    You must specify --task <id> or --task all. No default — downloading all
    six datasets at once can take significant time and disk space.
    """
    if task is None:
        raise click.UsageError("--task is required. Pass a task ID or 'all' to download every task.")
    task_ids = ALL_TASKS if task == "all" else [task]
    configs = load_task_configs(task_ids)

    failures = []
    for cfg in configs:
        click.echo(f"\n[{cfg.task_id}] Downloading {cfg.task_name}...")
        if dry_run:
            click.echo(f"  [dry-run] Would download {cfg.dataset_path} (config={cfg.dataset_config})")
            continue
        try:
            from datasets import load_dataset  # lazy import — not needed for dry-run
            out_dir = REPO_ROOT / "data" / "raw" / cfg.task_id
            out_dir.mkdir(parents=True, exist_ok=True)

            kwargs = {"path": cfg.dataset_path, "trust_remote_code": True}
            if cfg.dataset_config:
                kwargs["name"] = cfg.dataset_config

            ds = load_dataset(**kwargs)

            # Print row counts
            for split, dataset in ds.items():
                count = len(dataset)
                expected = EXPECTED_COUNTS.get(cfg.task_id, {}).get(split, 0)
                status = "OK" if count >= expected else f"WARNING: expected >= {expected}"
                click.echo(f"  {split}: {count:,} rows — {status}")

            # Save to disk
            ds.save_to_disk(str(out_dir))
            click.echo(f"  Saved to {out_dir}")

        except Exception as exc:
            click.echo(f"  ERROR: {exc}", err=True)
            failures.append((cfg.task_id, str(exc)))

    if failures:
        click.echo(f"\n{'='*50}")
        click.echo(f"FAILED ({len(failures)}):")
        for tid, err in failures:
            click.echo(f"  {tid}: {err}")
        sys.exit(1)
    else:
        click.echo(f"\nAll {'validated' if dry_run else 'downloaded'} successfully.")


if __name__ == "__main__":
    main()
