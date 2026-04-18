"""
Create tiny prepared datasets for local CPU testing without a full HuggingFace download.

Downloads a small slice of each task's real dataset (N_TRAIN train + N_TEST test rows)
and formats it identically to prepare_datasets.py. Output goes to data/prepared/{task}/
so train.py --tiny and eval_local.py --tiny find the files in the expected locations.

Usage:
    python scripts/make_tiny_data.py --task banking77
    python scripts/make_tiny_data.py --task all
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import click
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

ALL_TASKS = ["banking77", "cuad", "ledgar", "fpb", "medmcqa", "mbpp"]

N_TRAIN = 12   # training rows — small enough to train on CPU in ~2 minutes
N_TEST = 5     # test rows
N_FEW = 3      # few-shot examples


def load_prompt(task_id: str) -> dict:
    with open(REPO_ROOT / "prompts" / f"{task_id}.json") as f:
        return json.load(f)


def load_task_cfg(task_id: str) -> dict:
    with open(REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml") as f:
        return yaml.safe_load(f)


def write_jsonl(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    click.echo(f"  {path.relative_to(REPO_ROOT)}: {len(rows)} rows")


def get_label_names(ds, label_field: str, custom: list | None) -> list | None:
    if custom:
        return custom
    try:
        feat = ds.features.get(label_field)
        return feat.names if feat and hasattr(feat, "names") else None
    except Exception:
        return None


def fmt_train_rows(rows: list, prompt: dict, system: str, label_names: list | None) -> list:
    from prepare_datasets import format_user, format_assistant, to_chat
    return [
        to_chat(system, format_user(prompt, r), format_assistant(prompt, r, label_names))
        for r in rows
    ]


def fmt_test_prompts(rows: list, task_id: str, prompt: dict, system: str) -> list:
    from prepare_datasets import format_user
    out = []
    for i, r in enumerate(rows):
        d = {
            "id": f"{task_id}_test_{i:04d}",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": format_user(prompt, r)},
            ],
        }
        if task_id == "cuad":
            d["context"] = r.get("context", "")
        out.append(d)
    return out


def fmt_labels(rows: list, task_id: str, prompt: dict, label_names: list | None) -> list:
    from prepare_datasets import format_assistant
    return [
        {"id": f"{task_id}_test_{i:04d}", "label": format_assistant(prompt, r, label_names)}
        for i, r in enumerate(rows)
    ]


def make_task(task_id: str) -> None:
    from datasets import load_dataset
    from prepare_datasets import truncate_context

    cfg = load_task_cfg(task_id)
    prompt = load_prompt(task_id)
    system = prompt["system"]
    out_dir = REPO_ROOT / "data" / "prepared" / task_id
    ds_path = cfg["dataset_path"]
    ds_config = cfg.get("dataset_config")
    label_field = cfg.get("label_field", "label")

    click.echo(f"\n[{task_id}] Downloading tiny slice from HuggingFace...")

    if task_id == "fpb":
        # No predefined split — pull enough for all three subsets
        ds = load_dataset(ds_path, ds_config, split=f"train[:{N_TRAIN + N_TEST + N_FEW}]")
        rows = list(ds)
        label_names = get_label_names(ds, label_field, cfg.get("custom_label_names"))
        random.Random(42).shuffle(rows)
        train_rows = rows[:N_TRAIN]
        test_rows = rows[N_TRAIN:N_TRAIN + N_TEST]

    elif task_id == "medmcqa":
        train_ds = load_dataset(ds_path, ds_config, split=f"train[:{N_TRAIN}]")
        test_ds = load_dataset(ds_path, ds_config, split=f"validation[:{N_TEST}]")
        train_rows = list(train_ds)
        test_rows = list(test_ds)
        label_names = get_label_names(train_ds, label_field, cfg.get("custom_label_names"))

    elif task_id == "cuad":
        train_ds = load_dataset(ds_path, ds_config, split=f"train[:{N_TRAIN}]")
        test_ds = load_dataset(ds_path, ds_config, split=f"test[:{N_TEST}]")
        ctx_max = cfg.get("context_max_tokens", 1500)

        def flatten(raw_rows: list) -> list:
            out = []
            for r in raw_rows:
                ctx = truncate_context(r.get("context", ""), ctx_max)
                answers = r.get("answers", {})
                texts = answers.get("text", []) if isinstance(answers, dict) else []
                out.append({
                    "context": ctx,
                    "question": r.get("question", ""),
                    "answer_text": texts[0] if texts else "Not found.",
                    "_id": r.get("id", ""),
                })
            return out

        train_rows = flatten(list(train_ds))
        test_rows = flatten(list(test_ds))
        label_names = None

    else:
        # banking77, ledgar, mbpp — standard train/test splits
        test_split = cfg.get("test_split", "test")
        train_ds = load_dataset(ds_path, ds_config, split=f"train[:{N_TRAIN}]")
        test_ds = load_dataset(ds_path, ds_config, split=f"{test_split}[:{N_TEST}]")
        train_rows = list(train_ds)
        test_rows = list(test_ds)
        label_names = get_label_names(train_ds, label_field, cfg.get("custom_label_names"))

    few_shot = train_rows[:N_FEW]

    write_jsonl(fmt_train_rows(train_rows, prompt, system, label_names), out_dir / "train_500.jsonl")
    write_jsonl(fmt_train_rows(train_rows, prompt, system, label_names), out_dir / "train_full.jsonl")
    write_jsonl(fmt_test_prompts(test_rows, task_id, prompt, system), out_dir / "test.jsonl")
    write_jsonl(fmt_labels(test_rows, task_id, prompt, label_names), out_dir / "test_labels.jsonl")
    write_jsonl(fmt_train_rows(few_shot, prompt, system, label_names), out_dir / "few_shot_5.jsonl")
    write_jsonl(fmt_train_rows(train_rows, prompt, system, label_names), out_dir / "openai_sft_500.jsonl")

    click.echo(f"  [{task_id}] Done — {len(train_rows)} train, {len(test_rows)} test")


@click.command()
@click.option("--task", default="banking77", show_default=True, help="Task ID or 'all'")
def main(task: str) -> None:
    """Download tiny dataset slices for local CPU testing (no full download required)."""
    task_ids = ALL_TASKS if task == "all" else [task]
    failures = []
    for tid in task_ids:
        try:
            make_task(tid)
        except Exception as exc:
            click.echo(f"  ERROR [{tid}]: {exc}", err=True)
            import traceback
            traceback.print_exc()
            failures.append(tid)
    if failures:
        click.echo(f"\nFailed: {', '.join(failures)}")
        sys.exit(1)
    click.echo("\nTiny data ready. Run train.py --tiny and eval_local.py --tiny to test the pipeline.")


if __name__ == "__main__":
    main()
