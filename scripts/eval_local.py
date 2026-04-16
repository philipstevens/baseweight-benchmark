"""Evaluate fine-tuned local models via vLLM server."""
from __future__ import annotations

import asyncio
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

ALL_TASKS = ["ledgar", "cuad", "banking77", "fpb", "medmcqa", "mbpp"]
ALL_MODELS = ["qwen3-8b", "gemma3-4b", "phi4-mini"]

MAX_CONCURRENCY = 4
MAX_RETRIES = 3
VLLM_HOST = "http://localhost:8000"
VLLM_HEALTH_TIMEOUT = 300  # seconds to wait for vLLM to be ready
VLLM_HEALTH_INTERVAL = 5


class TaskConfig(BaseModel):
    task_id: str
    max_output_tokens: int
    task_type: str


class ModelConfig(BaseModel):
    model_id: str
    model_short: str
    max_seq_length: int = 2048
    fallback_model_id: Optional[str] = None


def load_task_config(task_id: str) -> TaskConfig:
    path = REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return TaskConfig(**{k: v for k, v in data.items() if k in TaskConfig.model_fields})


def load_model_config(model_id: str) -> ModelConfig:
    path = REPO_ROOT / "configs" / "training" / f"{model_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return ModelConfig(**{k: v for k, v in data.items() if k in ModelConfig.model_fields})


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_messages(prompt_row: dict, few_shot: list[dict], condition: str) -> list[dict]:
    """Build message list, optionally prepending few-shot examples."""
    base = prompt_row["messages"]
    if condition == "5-shot" and few_shot:
        system = base[0]
        user = base[1]
        shots = []
        for ex in few_shot:
            msgs = ex.get("messages", [])
            if len(msgs) >= 3:
                shots.append(msgs[1])
                shots.append(msgs[2])
        return [system] + shots + [user]
    return base


def start_vllm_server(base_model: str, adapter_path: Optional[Path], max_seq_length: int) -> subprocess.Popen:
    """Start vLLM server process."""
    cmd = [
        "vllm", "serve", base_model,
        "--dtype", "bfloat16",
        "--max-model-len", str(max_seq_length),
        "--gpu-memory-utilization", "0.9",
        "--port", "8000",
    ]
    if adapter_path and adapter_path.exists():
        cmd += [
            "--enable-lora",
            "--lora-modules", f"adapter={adapter_path}",
        ]

    click.echo(f"  Starting vLLM: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    return proc


def stop_vllm_server(proc: subprocess.Popen) -> None:
    """Gracefully stop vLLM server."""
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    click.echo("  vLLM server stopped.")


async def wait_for_vllm(timeout: int = VLLM_HEALTH_TIMEOUT) -> bool:
    """Poll /health endpoint until server is ready."""
    import aiohttp
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{VLLM_HOST}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        click.echo("  vLLM server is ready.")
                        return True
        except Exception:
            pass
        await asyncio.sleep(VLLM_HEALTH_INTERVAL)
    return False


async def call_vllm(
    model_name: str,
    messages: list[dict],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, int, int, float, float]:
    """Call vLLM via OpenAI-compatible API. Returns (text, in_tok, out_tok, latency_ms, ttft_ms)."""
    import aiohttp

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
    }

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                ttft_ms = 0.0
                chunks = []

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{VLLM_HOST}/v1/chat/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300),
                    ) as resp:
                        resp.raise_for_status()
                        first_token = True
                        async for raw_line in resp.content:
                            line = raw_line.decode("utf-8").strip()
                            if not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content and first_token:
                                ttft_ms = (time.time() - t0) * 1000
                                first_token = False
                            if content:
                                chunks.append(content)

                latency_ms = (time.time() - t0) * 1000
                text = "".join(chunks)
                # Token counts not available from streaming; estimate
                in_tok = 0
                out_tok = 0
                return text, in_tok, out_tok, latency_ms, ttft_ms

            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
    return "", 0, 0, 0, 0


def run_mbpp_tests(code: str, test_list: list[str], timeout: int = 5) -> tuple[bool, str]:
    """Run MBPP test assertions in a subprocess sandbox. Returns (passed, error_msg)."""
    full_code = code + "\n" + "\n".join(test_list)
    script = f"""
import resource, sys
resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
exec(compile({repr(full_code)}, '<mbpp>', 'exec'))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout + 2,
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr or result.stdout).strip()[:500]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:
        return False, str(exc)


def extract_code(text: str) -> str:
    """Extract Python code from model output (strips markdown fences)."""
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        return text[start:end].strip() if end != -1 else text[start:].strip()
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        return text[start:end].strip() if end != -1 else text[start:].strip()
    return text.strip()


async def run_eval(
    model_cfg: ModelConfig,
    task_id: str,
    condition: str,
    task_cfg: TaskConfig,
    adapter_path: Optional[Path],
    dry_run: bool,
) -> None:
    prepared = REPO_ROOT / "data" / "prepared" / task_id
    test_path = prepared / "test.jsonl"
    few_shot_path = prepared / "few_shot_5.jsonl"

    if not test_path.exists():
        click.echo(f"  SKIP [{model_cfg.model_short}/{task_id}/{condition}]: test.jsonl not found")
        return

    test_rows = load_jsonl(test_path)
    few_shot = load_jsonl(few_shot_path) if few_shot_path.exists() else []

    out_path = (
        REPO_ROOT / "results" / "predictions"
        / model_cfg.model_short / task_id / f"{condition}.jsonl"
    )

    if dry_run:
        click.echo(f"  [dry-run] Would eval {model_cfg.model_short} on {task_id}/{condition} ({len(test_rows)} examples)")
        return

    if out_path.exists():
        click.echo(f"  SKIP [{model_cfg.model_short}/{task_id}/{condition}]: already exists")
        return

    # For adapter conditions, verify adapter exists
    model_name = "adapter" if adapter_path and adapter_path.exists() else model_cfg.model_id

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # Load MBPP test lists if needed
    mbpp_tests: dict[str, list[str]] = {}
    if task_id == "mbpp":
        raw_path = REPO_ROOT / "data" / "raw" / "mbpp"
        if (raw_path / "dataset_dict.json").exists() or raw_path.exists():
            try:
                from datasets import load_from_disk
                ds = load_from_disk(str(raw_path))
                split = "test" if "test" in ds else list(ds.keys())[0]
                for i, row in enumerate(ds[split]):
                    key = f"mbpp_test_{i:04d}"
                    mbpp_tests[key] = row.get("test_list", [])
            except Exception:
                pass

    async def process_row(row: dict) -> dict:
        msgs = build_messages(row, few_shot, condition)
        ground_truth = row.get("label", "")
        row_id = row.get("id", "")
        try:
            text, in_tok, out_tok, lat, ttft = await call_vllm(
                model_name, msgs, task_cfg.max_output_tokens, semaphore
            )
        except Exception as exc:
            text, in_tok, out_tok, lat, ttft = f"ERROR: {exc}", 0, 0, 0, 0

        result: dict = {
            "id": row_id,
            "model": model_cfg.model_short,
            "condition": condition,
            "input": msgs[-1]["content"] if msgs else "",
            "output": text,
            "ground_truth": ground_truth,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_ms": lat,
            "ttft_ms": ttft,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

        # MBPP: run tests
        if task_id == "mbpp" and not text.startswith("ERROR:"):
            code = extract_code(text)
            tests = mbpp_tests.get(row_id, [])
            if tests:
                passed, err = run_mbpp_tests(code, tests)
                result["mbpp_passed"] = passed
                result["mbpp_error"] = err
            else:
                result["mbpp_passed"] = None
                result["mbpp_error"] = "no tests found"

        return result

    click.echo(f"  Evaluating {model_cfg.model_short}/{task_id}/{condition} ({len(test_rows)} examples)...")
    tasks_list = [process_row(r) for r in test_rows]
    from tqdm.asyncio import tqdm
    predictions = await tqdm.gather(*tasks_list, desc=f"{model_cfg.model_short}/{task_id}")

    write_jsonl(predictions, out_path)
    click.echo(f"  Saved {len(predictions)} predictions to {out_path.relative_to(REPO_ROOT)}")


@click.command()
@click.option("--model", required=True, help="Model ID (qwen3-8b|gemma3-4b|phi4-mini)")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--condition", default="all", help="zero-shot|5-shot|lora-500|lora-full|all")
@click.option("--dry-run", is_flag=True, help="Validate without running inference")
def main(model: str, task: str, condition: str, dry_run: bool) -> None:
    """Evaluate local fine-tuned models via vLLM server."""
    model_ids = ALL_MODELS if model == "all" else [model]
    task_ids = ALL_TASKS if task == "all" else [task]

    failures = []

    for mid in model_ids:
        model_cfg = load_model_config(mid)

        # Determine conditions: zero-shot/5-shot use base model; lora-* use adapter
        if condition == "all":
            conditions = ["zero-shot", "5-shot", "lora-500", "lora-full"]
        else:
            conditions = [condition]

        # Group conditions by whether they need an adapter
        base_conditions = [c for c in conditions if c in ("zero-shot", "5-shot")]
        lora_conditions = [c for c in conditions if c.startswith("lora-")]

        for tid in task_ids:
            try:
                task_cfg = load_task_config(tid)
            except Exception as exc:
                click.echo(f"  ERROR: could not load task config for {tid}: {exc}", err=True)
                failures.append((f"{mid}/{tid}", str(exc)))
                continue

            # --- Base model (zero-shot, 5-shot) ---
            if base_conditions:
                if dry_run:
                    for cond in base_conditions:
                        asyncio.run(run_eval(model_cfg, tid, cond, task_cfg, None, dry_run=True))
                else:
                    proc = start_vllm_server(model_cfg.model_id, None, model_cfg.max_seq_length)
                    try:
                        ready = asyncio.run(wait_for_vllm())
                        if not ready:
                            raise RuntimeError("vLLM server did not become ready in time")
                        for cond in base_conditions:
                            try:
                                asyncio.run(run_eval(model_cfg, tid, cond, task_cfg, None, dry_run=False))
                            except Exception as exc:
                                click.echo(f"  ERROR [{mid}/{tid}/{cond}]: {exc}", err=True)
                                failures.append((f"{mid}/{tid}/{cond}", str(exc)))
                    finally:
                        stop_vllm_server(proc)

            # --- LoRA adapter conditions ---
            for cond in lora_conditions:
                adapter_path = (
                    REPO_ROOT / "results" / "adapters" / model_cfg.model_short / tid / cond
                )
                if not adapter_path.exists():
                    click.echo(f"  SKIP [{mid}/{tid}/{cond}]: adapter not found at {adapter_path}")
                    continue

                if dry_run:
                    asyncio.run(run_eval(model_cfg, tid, cond, task_cfg, adapter_path, dry_run=True))
                    continue

                proc = start_vllm_server(model_cfg.model_id, adapter_path, model_cfg.max_seq_length)
                try:
                    ready = asyncio.run(wait_for_vllm())
                    if not ready:
                        raise RuntimeError("vLLM server did not become ready in time")
                    asyncio.run(run_eval(model_cfg, tid, cond, task_cfg, adapter_path, dry_run=False))
                except Exception as exc:
                    click.echo(f"  ERROR [{mid}/{tid}/{cond}]: {exc}", err=True)
                    failures.append((f"{mid}/{tid}/{cond}", str(exc)))
                finally:
                    stop_vllm_server(proc)

    if failures:
        click.echo(f"\nFAILED ({len(failures)}):")
        for key, err in failures:
            click.echo(f"  {key}: {err}")
        sys.exit(1)
    click.echo("\nAll local evaluations completed.")


if __name__ == "__main__":
    main()
