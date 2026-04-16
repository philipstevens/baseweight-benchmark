"""Evaluate frontier API models on benchmark tasks."""
from __future__ import annotations

import asyncio
import json
import os
import sys
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
ALL_API_MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "claude-sonnet", "gemini-flash"]
OPENAI_MODELS = {"gpt-4.1": "gpt-4.1", "gpt-4.1-mini": "gpt-4.1-mini", "gpt-4.1-nano": "gpt-4.1-nano", "gpt-4.1-mini-sft": None}
ANTHROPIC_MODELS = {"claude-sonnet": "claude-sonnet-4-6-20250514"}
GOOGLE_MODELS = {"gemini-flash": "gemini-2.5-flash"}

MAX_CONCURRENCY = 5
MAX_RETRIES = 5


class TaskConfig(BaseModel):
    task_id: str
    max_output_tokens: int
    task_type: str


def load_task_config(task_id: str) -> TaskConfig:
    path = REPO_ROOT / "configs" / "tasks" / f"{task_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return TaskConfig(**{k: v for k, v in data.items() if k in TaskConfig.model_fields})


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
                shots.append(msgs[1])  # user
                shots.append(msgs[2])  # assistant
        return [system] + shots + [user]
    return base


async def call_openai(client, model_str: str, messages: list[dict], max_tokens: int, semaphore: asyncio.Semaphore) -> tuple[str, int, int, float]:
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                resp = await client.chat.completions.create(
                    model=model_str, messages=messages, temperature=0, max_tokens=max_tokens,
                )
                latency_ms = int((time.time() - t0) * 1000)
                text = resp.choices[0].message.content or ""
                return text, resp.usage.prompt_tokens, resp.usage.completion_tokens, latency_ms
            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = 2 ** attempt
                await asyncio.sleep(wait)
    return "", 0, 0, 0


async def call_anthropic(client, model_str: str, messages: list[dict], max_tokens: int, semaphore: asyncio.Semaphore) -> tuple[str, int, int, float]:
    async with semaphore:
        # Extract system message
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        non_system = [m for m in messages if m["role"] != "system"]
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                resp = await client.messages.create(
                    model=model_str, system=system, messages=non_system,
                    temperature=0, max_tokens=max_tokens,
                )
                latency_ms = int((time.time() - t0) * 1000)
                text = resp.content[0].text if resp.content else ""
                return text, resp.usage.input_tokens, resp.usage.output_tokens, latency_ms
            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
    return "", 0, 0, 0


async def call_gemini(client, model_str: str, messages: list[dict], max_tokens: int, semaphore: asyncio.Semaphore) -> tuple[str, int, int, float]:
    import google.generativeai as genai
    async with semaphore:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        # Convert to Gemini format
        history = []
        for m in user_msgs[:-1]:
            history.append({"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]})
        last_user = user_msgs[-1]["content"] if user_msgs else ""
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                model_obj = genai.GenerativeModel(model_str, system_instruction=system)
                chat = model_obj.start_chat(history=history)
                resp = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: chat.send_message(last_user, generation_config=genai.GenerationConfig(temperature=0, max_output_tokens=max_tokens))
                )
                latency_ms = int((time.time() - t0) * 1000)
                text = resp.text or ""
                in_tok = resp.usage_metadata.prompt_token_count if resp.usage_metadata else 0
                out_tok = resp.usage_metadata.candidates_token_count if resp.usage_metadata else 0
                return text, in_tok, out_tok, latency_ms
            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
    return "", 0, 0, 0


async def run_eval(model_id: str, task_id: str, condition: str, task_cfg: TaskConfig, dry_run: bool) -> None:
    prepared = REPO_ROOT / "data" / "prepared" / task_id
    test_path = prepared / "test.jsonl"
    few_shot_path = prepared / "few_shot_5.jsonl"

    if not test_path.exists():
        click.echo(f"  SKIP [{model_id}/{task_id}/{condition}]: test.jsonl not found")
        return

    test_rows = load_jsonl(test_path)
    few_shot = load_jsonl(few_shot_path) if few_shot_path.exists() else []

    if dry_run:
        click.echo(f"  [dry-run] Would eval {model_id} on {task_id}/{condition} ({len(test_rows)} examples)")
        return

    out_path = REPO_ROOT / "results" / "predictions" / model_id / task_id / f"{condition}.jsonl"
    if out_path.exists():
        click.echo(f"  SKIP [{model_id}/{task_id}/{condition}]: already exists")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    predictions = []

    # Determine which client to use
    if model_id in OPENAI_MODELS or model_id == "gpt-4.1-mini-sft":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        model_str = OPENAI_MODELS.get(model_id, model_id)
    elif model_id in ANTHROPIC_MODELS:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        model_str = ANTHROPIC_MODELS[model_id]
    elif model_id in GOOGLE_MODELS:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        client = None
        model_str = GOOGLE_MODELS[model_id]
    else:
        raise ValueError(f"Unknown model: {model_id}")

    async def process_row(row: dict) -> dict:
        msgs = build_messages(row, few_shot, condition)
        ground_truth = row.get("label", "")
        try:
            if model_id in ANTHROPIC_MODELS:
                text, in_tok, out_tok, lat = await call_anthropic(client, model_str, msgs, task_cfg.max_output_tokens, semaphore)
            elif model_id in GOOGLE_MODELS:
                text, in_tok, out_tok, lat = await call_gemini(client, model_str, msgs, task_cfg.max_output_tokens, semaphore)
            else:
                text, in_tok, out_tok, lat = await call_openai(client, model_str, msgs, task_cfg.max_output_tokens, semaphore)
        except Exception as exc:
            text, in_tok, out_tok, lat = f"ERROR: {exc}", 0, 0, 0

        return {
            "id": row.get("id", ""),
            "model": model_id,
            "condition": condition,
            "input": msgs[-1]["content"] if msgs else "",
            "output": text,
            "ground_truth": ground_truth,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_ms": lat,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    click.echo(f"  Evaluating {model_id}/{task_id}/{condition} ({len(test_rows)} examples)...")
    tasks_list = [process_row(r) for r in test_rows]
    from tqdm.asyncio import tqdm
    predictions = await tqdm.gather(*tasks_list, desc=f"{model_id}/{task_id}")

    write_jsonl(predictions, out_path)
    click.echo(f"  Saved {len(predictions)} predictions to {out_path.relative_to(REPO_ROOT)}")


async def run_sft(task_id: str, task_cfg: TaskConfig, dry_run: bool) -> None:
    """Upload training data, create fine-tuning job, evaluate fine-tuned model."""
    from openai import AsyncOpenAI, OpenAI
    import time as _time

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    prepared = REPO_ROOT / "data" / "prepared" / task_id
    sft_path = prepared / "openai_sft_500.jsonl"

    if not sft_path.exists():
        click.echo(f"  SKIP [gpt-4.1-mini-sft/{task_id}]: {sft_path} not found")
        return

    if dry_run:
        click.echo(f"  [dry-run] Would upload {sft_path} and create fine-tuning job")
        return

    # Check if we already have a fine-tuned model saved
    meta_path = REPO_ROOT / "results" / "training" / "gpt-4.1-mini-sft" / task_id / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        ft_model = meta.get("ft_model_id")
    else:
        click.echo(f"  Uploading training file for {task_id}...")
        with open(sft_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="fine-tune")

        job = client.fine_tuning.jobs.create(
            training_file=file_obj.id,
            model="gpt-4.1-mini",
        )
        click.echo(f"  Fine-tuning job created: {job.id}. Waiting for completion...")

        while job.status not in ("succeeded", "failed", "cancelled"):
            _time.sleep(60)
            job = client.fine_tuning.jobs.retrieve(job.id)
            click.echo(f"  Status: {job.status}")

        if job.status != "succeeded":
            raise RuntimeError(f"Fine-tuning job failed: {job.status}")

        ft_model = job.fine_tuned_model
        training_cost = (job.trained_tokens or 0) * 0.80 / 1_000_000
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump({"ft_model_id": ft_model, "job_id": job.id, "trained_tokens": job.trained_tokens, "training_cost": training_cost}, f)
        click.echo(f"  Fine-tuned model: {ft_model}, cost: ${training_cost:.3f}")

    # Evaluate the fine-tuned model
    click.echo(f"  Evaluating fine-tuned model {ft_model} on {task_id}...")
    OPENAI_MODELS["gpt-4.1-mini-sft"] = ft_model
    await run_eval("gpt-4.1-mini-sft", task_id, "api-sft-500", task_cfg, dry_run)


@click.command()
@click.option("--model", default="all", help="Model ID or 'all'")
@click.option("--task", default="all", help="Task ID or 'all'")
@click.option("--condition", default="all", help="zero-shot|5-shot|api-sft-500|all")
@click.option("--dry-run", is_flag=True)
def main(model: str, task: str, condition: str, dry_run: bool) -> None:
    """Evaluate frontier API models (prompted and SFT)."""
    # Validate API keys
    if not dry_run:
        for key, var in [("OpenAI", "OPENAI_API_KEY"), ("Anthropic", "ANTHROPIC_API_KEY"), ("Google", "GOOGLE_API_KEY")]:
            if not os.environ.get(var):
                click.echo(f"  WARNING: {var} not set — {key} models will fail", err=True)

    model_ids = ALL_API_MODELS if model == "all" else [model]
    task_ids = ALL_TASKS if task == "all" else [task]
    conditions = ["zero-shot", "5-shot"] if condition == "all" else [condition]

    failures = []

    async def run_all() -> None:
        for mid in model_ids:
            for tid in task_ids:
                try:
                    task_cfg = load_task_config(tid)
                    if mid == "gpt-4.1-mini-sft":
                        await run_sft(tid, task_cfg, dry_run)
                    else:
                        for cond in conditions:
                            await run_eval(mid, tid, cond, task_cfg, dry_run)
                except Exception as exc:
                    click.echo(f"  ERROR [{mid}/{tid}]: {exc}", err=True)
                    import traceback; traceback.print_exc()
                    failures.append((f"{mid}/{tid}", str(exc)))

    asyncio.run(run_all())

    if failures:
        click.echo(f"\nFAILED ({len(failures)}):")
        for key, err in failures:
            click.echo(f"  {key}: {err}")
        sys.exit(1)
    click.echo("\nAll API evaluations completed.")


if __name__ == "__main__":
    main()
