"""Sync all run artifacts to HuggingFace for RunPod persistence.

Run this after any training or eval step to ensure nothing is lost if the pod
terminates. Safe to run multiple times — existing files are skipped.

Usage:
    python scripts/sync_artifacts.py                  # sync everything
    python scripts/sync_artifacts.py --what adapters  # adapters only
    python scripts/sync_artifacts.py --what data      # prepared datasets only
    python scripts/sync_artifacts.py --what all       # everything
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

HF_ORG = "baseweight"  # update to your HF username/org
DATASET_REPO = f"{HF_ORG}/baseweight-benchmark-data"
PREDICTIONS_REPO = f"{HF_ORG}/baseweight-benchmark-predictions"


def get_api():
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        click.echo("ERROR: HF_TOKEN not set in .env", err=True)
        sys.exit(1)
    return HfApi(token=token)


def ensure_repo(api, repo_id: str, repo_type: str) -> None:
    from huggingface_hub import create_repo
    try:
        create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=False)
    except Exception as exc:
        click.echo(f"  WARNING: could not create {repo_id}: {exc}", err=True)


def upload_file_safe(api, local_path: Path, remote_path: str, repo_id: str, repo_type: str) -> bool:
    """Upload a single file. Returns True on success."""
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"sync: {remote_path}",
        )
        return True
    except Exception as exc:
        click.echo(f"  ERROR uploading {remote_path}: {exc}", err=True)
        return False


def sync_prepared_data(api, dry_run: bool) -> int:
    """Upload all prepared JSONL files to HuggingFace dataset repo."""
    prepared_root = REPO_ROOT / "data" / "prepared"
    if not prepared_root.exists():
        click.echo("  No prepared data found — skipping")
        return 0

    if not dry_run:
        ensure_repo(api, DATASET_REPO, "dataset")

    files = list(prepared_root.rglob("*.jsonl"))
    count = 0
    for f in files:
        remote = f"prepared/{f.relative_to(prepared_root)}"
        if dry_run:
            click.echo(f"  [dry-run] Would upload {remote}")
            count += 1
        else:
            if upload_file_safe(api, f, remote, DATASET_REPO, "dataset"):
                count += 1

    click.echo(f"  Prepared data: {count}/{len(files)} files synced to {DATASET_REPO}")
    return count


def sync_adapters(api, dry_run: bool) -> int:
    """Upload LoRA adapter weights to HuggingFace model repos."""
    adapter_root = REPO_ROOT / "results" / "adapters"
    if not adapter_root.exists():
        click.echo("  No adapters found — skipping")
        return 0

    count = 0
    for model_dir in adapter_root.iterdir():
        if not model_dir.is_dir():
            continue
        for task_dir in model_dir.iterdir():
            if not task_dir.is_dir():
                continue
            for condition_dir in task_dir.iterdir():
                if not condition_dir.is_dir():
                    continue
                model_short = model_dir.name
                task_id = task_dir.name
                condition = condition_dir.name
                repo_id = f"{HF_ORG}/{model_short}-{task_id}-{condition}"

                if dry_run:
                    files = list(condition_dir.rglob("*"))
                    click.echo(f"  [dry-run] Would upload {len(files)} files to {repo_id}")
                    count += 1
                    continue

                try:
                    from huggingface_hub import create_repo
                    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
                    api.upload_folder(
                        folder_path=str(condition_dir),
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=f"sync: {model_short}/{task_id}/{condition} adapter",
                        ignore_patterns=["README.md"],
                    )
                    click.echo(f"  Synced adapter → {repo_id}")
                    count += 1
                except Exception as exc:
                    click.echo(f"  ERROR syncing {repo_id}: {exc}", err=True)

    click.echo(f"  Adapters: {count} repos synced")
    return count


def sync_predictions(api, dry_run: bool) -> int:
    """Upload prediction JSONL files to HuggingFace dataset repo."""
    pred_root = REPO_ROOT / "results" / "predictions"
    if not pred_root.exists():
        click.echo("  No predictions found — skipping")
        return 0

    if not dry_run:
        ensure_repo(api, PREDICTIONS_REPO, "dataset")

    files = list(pred_root.rglob("*.jsonl"))
    count = 0
    for f in files:
        remote = f"predictions/{f.relative_to(pred_root)}"
        if dry_run:
            click.echo(f"  [dry-run] Would upload {remote}")
            count += 1
        else:
            if upload_file_safe(api, f, remote, PREDICTIONS_REPO, "dataset"):
                count += 1

    click.echo(f"  Predictions: {count}/{len(files)} files synced to {PREDICTIONS_REPO}")
    return count


def sync_summaries(api, dry_run: bool) -> int:
    """Upload summary JSON files to HuggingFace dataset repo."""
    summary_root = REPO_ROOT / "results" / "summaries"
    if not summary_root.exists():
        click.echo("  No summaries found — skipping")
        return 0

    if not dry_run:
        ensure_repo(api, PREDICTIONS_REPO, "dataset")

    files = list(summary_root.rglob("*.json"))
    count = 0
    for f in files:
        remote = f"summaries/{f.relative_to(summary_root)}"
        if dry_run:
            click.echo(f"  [dry-run] Would upload {remote}")
            count += 1
        else:
            if upload_file_safe(api, f, remote, PREDICTIONS_REPO, "dataset"):
                count += 1

    click.echo(f"  Summaries: {count}/{len(files)} files synced")
    return count


def sync_training_metadata(api, dry_run: bool) -> int:
    """Upload training metadata.json files."""
    training_root = REPO_ROOT / "results" / "training"
    if not training_root.exists():
        click.echo("  No training metadata found — skipping")
        return 0

    if not dry_run:
        ensure_repo(api, PREDICTIONS_REPO, "dataset")

    files = list(training_root.rglob("metadata.json"))
    count = 0
    for f in files:
        remote = f"training/{f.relative_to(training_root)}"
        if dry_run:
            click.echo(f"  [dry-run] Would upload {remote}")
            count += 1
        else:
            if upload_file_safe(api, f, remote, PREDICTIONS_REPO, "dataset"):
                count += 1

    click.echo(f"  Training metadata: {count}/{len(files)} files synced")
    return count


@click.command()
@click.option(
    "--what",
    default="all",
    type=click.Choice(["all", "data", "adapters", "predictions", "summaries"]),
    help="What to sync",
)
@click.option("--dry-run", is_flag=True)
def main(what: str, dry_run: bool) -> None:
    """Sync all run artifacts to HuggingFace (RunPod persistence)."""
    if dry_run:
        click.echo("  [dry-run mode — no uploads will occur]\n")
        api = None
    else:
        api = get_api()

    if what in ("all", "data"):
        sync_prepared_data(api, dry_run)

    if what in ("all", "adapters"):
        sync_adapters(api, dry_run)

    if what in ("all", "predictions"):
        sync_predictions(api, dry_run)

    if what in ("all", "summaries"):
        sync_summaries(api, dry_run)
        sync_training_metadata(api, dry_run)

    click.echo("\nSync complete.")


if __name__ == "__main__":
    main()
