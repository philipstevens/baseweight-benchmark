# Baseweight Benchmark

Fine-tuned open-source models vs. frontier APIs on vertical SaaS tasks.

Six tasks. Three fine-tuned models. Five frontier APIs. Full cost analysis.

**Results:** [baseweight.co/benchmark](https://baseweight.co/benchmark)
**Methodology:** [baseweight.co/methodology](https://baseweight.co/methodology)
**Weights:** [HuggingFace collection](https://huggingface.co/collections/baseweight/baseweight-benchmark)

## What this is

A reproducible benchmark comparing QLoRA-fine-tuned open-source models (Qwen3-8B, Gemma 3-4B, Phi-4-mini) against frontier APIs (GPT-4.1, Claude Sonnet 4.6, Gemini 2.5 Flash) on six production tasks: legal document classification, contract clause extraction, customer support routing, financial sentiment, medical QA, and code generation.

Every result includes accuracy, cost per correct prediction, latency, and an error analysis.

## Reproduce

Requires: NVIDIA A100 80GB GPU, Python 3.11+, API keys for OpenAI/Anthropic/Google. Total cost: ~$300.

```bash
git clone https://github.com/baseweight/baseweight-benchmark.git
cd baseweight-benchmark
bash scripts/setup.sh
cp .env.example .env  # add API keys
python scripts/download_data.py
python scripts/prepare_datasets.py
```

### Train

```bash
python scripts/train.py --model qwen3-8b --task all
python scripts/train.py --model gemma3-4b --task all
python scripts/train.py --model phi4-mini --task all
python scripts/train.py --model qwen3-8b --task all --efficiency-curve
```

### Evaluate

```bash
python scripts/eval_local.py --task all
python scripts/eval_api.py --task all
python scripts/classify_errors.py
python scripts/generate_dashboard_data.py
```

## Tasks

| Task | Dataset | Metric |
|------|---------|--------|
| Legal document classification | LEDGAR (LexGLUE) | Weighted F1 |
| Contract clause extraction | CUAD | Token F1 |
| Customer support routing | BANKING77 | Weighted F1 |
| Financial sentiment | FinancialPhraseBank | Weighted F1 |
| Medical QA | MedMCQA | Accuracy |
| Code generation | MBPP | Pass@1 |

## License

Code: MIT. Model adapters: inherit base model license. Datasets: see individual licenses.
