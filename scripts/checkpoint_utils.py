"""Fault-tolerant checkpoint and recovery utilities for remote GPU training.

Network volume layout (under NETWORK_VOLUME/checkpoints/<model>/<task>/<condition>/):
  checkpoint-N/      HF Trainer intermediate checkpoints (one per epoch)
  train_state.json   Lightweight state: epoch, step, loss, status
  adapter/           Final adapter copy (written atomically on job completion)

Partial eval layout:
  results/predictions/<model>/<task>/<condition>.jsonl.partial
    Written row-by-row as inference completes; renamed to .jsonl on completion.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

# Set NETWORK_VOLUME to your remote GPU network volume mount point.
NETWORK_VOLUME = Path(os.environ.get("NETWORK_VOLUME", "/workspace"))


# ── Atomic I/O ────────────────────────────────────────────────────────────────

def atomic_write_json(data: dict, path: Path) -> None:
    """Write JSON atomically: write to .tmp, fsync, os.replace (POSIX atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(row: dict, path: Path) -> None:
    """Append one JSONL row and fsync.

    Safe in a single-threaded asyncio event loop: f.write() does not yield,
    so concurrent coroutines cannot interleave within a single append call.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def finalize_partial(partial_path: Path, final_path: Path) -> None:
    """Atomically promote a completed partial file to its final name."""
    os.replace(partial_path, final_path)


# ── Eval partial-progress helpers ─────────────────────────────────────────────

def partial_path(out_path: Path) -> Path:
    """Return the .partial sibling of a predictions output path."""
    return out_path.with_name(out_path.name + ".partial")


def load_partial_ids(pp: Path) -> set[str]:
    """Return set of row IDs already written to a partial predictions file."""
    if not pp.exists():
        return set()
    ids: set[str] = set()
    with open(pp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row_id = json.loads(line).get("id", "")
                if row_id:
                    ids.add(row_id)
            except json.JSONDecodeError:
                continue
    return ids


# ── Training checkpoint helpers ───────────────────────────────────────────────

def checkpoint_dir(model_short: str, task_id: str, condition: str) -> Path:
    return NETWORK_VOLUME / "checkpoints" / model_short / task_id / condition


def find_hf_resume_checkpoint(
    model_short: str, task_id: str, condition: str
) -> Optional[Path]:
    """Return the latest HF Trainer checkpoint directory, or None."""
    ckpt_dir = checkpoint_dir(model_short, task_id, condition)
    if not ckpt_dir.exists():
        return None
    candidates = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    return candidates[-1] if candidates else None


def save_train_state(
    model_short: str, task_id: str, condition: str, state: dict
) -> None:
    """Atomically write training state to the network volume."""
    path = checkpoint_dir(model_short, task_id, condition) / "train_state.json"
    atomic_write_json({**state, "saved_at": time.time()}, path)


def load_train_state(
    model_short: str, task_id: str, condition: str
) -> Optional[dict]:
    path = checkpoint_dir(model_short, task_id, condition) / "train_state.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
