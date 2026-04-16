#!/usr/bin/env bash
set -euo pipefail

echo "=== Baseweight Benchmark Setup ==="

# Create conda environment
if conda env list | grep -q "baseweight-bench"; then
    echo "Environment baseweight-bench already exists. Activating..."
else
    echo "Creating conda environment baseweight-bench (Python 3.11)..."
    conda create -n baseweight-bench python=3.11 -y
fi

# Activate
eval "$(conda shell.bash hook)"
conda activate baseweight-bench

echo "Installing requirements..."
pip install -r requirements.txt

echo "Installing Unsloth..."
pip install "unsloth[cu124]"

echo "Installing vLLM..."
pip install vllm

echo "Verifying CUDA..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — check your GPU and CUDA installation'; print('CUDA OK:', torch.version.cuda)"

echo "GPU info:"
python -c "import torch; print(torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')"

echo ""
echo "=== Setup complete ==="
echo "Next: cp .env.example .env  # then add your API keys"
