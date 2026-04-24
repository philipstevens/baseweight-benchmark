"""Download raw datasets from HuggingFace for all benchmark tasks."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from pydantic import BaseModel
import yaml

# Expected minimum row counts for sanity checks (split: min_count)
EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "banking77": {"train": 8000,  "test": 2000},
    "cuad":      {"train": 10000, "test": 4000},
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

TINY_TRAIN = 12
TINY_TEST = 5


def download_task(cfg: TaskConfig, dry_run: bool, tiny: bool = False) -> None:
    click.echo(f"\n[{cfg.task_id}] Downloading {cfg.task_name}...")
    if dry_run:
        click.echo(f"  [dry-run] Would download {cfg.dataset_path} (config={cfg.dataset_config})")
        return

    from datasets import load_dataset, DatasetDict  # lazy import
    out_dir = REPO_ROOT / "data" / "raw" / cfg.task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs: dict = {}
    if cfg.dataset_config:
        load_kwargs["name"] = cfg.dataset_config

    if tiny:
        loaded = {}
        split_errors: list[str] = []
        for split in ("train", "test", "validation"):
            limit = TINY_TRAIN if split == "train" else TINY_TEST
            try:
                ds_split = load_dataset(cfg.dataset_path, split=f"{split}[:{limit}]", **load_kwargs)
                loaded[split] = ds_split
                click.echo(f"  {split}: {len(ds_split)} rows (tiny)")
            except Exception as e:
                split_errors.append(f"{split}: {e}")
        if not loaded:
            for err in split_errors:
                click.echo(f"  {err}", err=True)
            raise RuntimeError(f"No valid splits found for {cfg.dataset_path}")
        ds = DatasetDict(loaded)
    else:
        ds = load_dataset(cfg.dataset_path, **load_kwargs)
        for split, dataset in ds.items():
            count = len(dataset)
            expected = EXPECTED_COUNTS.get(cfg.task_id, {}).get(split, 0)
            status = "OK" if count >= expected else f"WARNING: expected >= {expected}"
            click.echo(f"  {split}: {count:,} rows — {status}")

    ds.save_to_disk(str(out_dir))
    click.echo(f"  Saved to {out_dir}")


@click.command()
@click.option("--task", default=None, help="Task ID to download (required; use 'all' to download every task)")
@click.option("--dry-run", is_flag=True, help="Validate config without downloading")
@click.option("--tiny", is_flag=True, help=f"Download only {TINY_TRAIN} train + {TINY_TEST} test rows for local CPU testing")
def main(task: str, dry_run: bool, tiny: bool) -> None:
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
        try:
            download_task(cfg, dry_run, tiny=tiny)
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
