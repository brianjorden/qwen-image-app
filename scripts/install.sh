#!/bin/bash
# Installation script for Qwen-Image App
# Rebuilds venv and installs all dependencies

set -euo pipefail

echo "=========================================="""
echo "Qwen-Image App Installation"
echo "=========================================="

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found. Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Cleaning up old environment..."
rm -rf app/__pycache__
rm -rf src/__pycache__
rm -rf .venv

echo "Creating fresh virtual environment..."
uv venv
source .venv/bin/activate

echo "Installing core dependencies..."
uv pip install --upgrade pip wheel setuptools packaging

echo "Installing PyTorch with CUDA 12.8..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "Installing Diffusers from latest source..."
uv pip install --upgrade "git+https://github.com/huggingface/diffusers.git"

echo "Installing Transformers and Accelerate..."
uv pip install transformers accelerate

echo "Installing UI and utility packages..."
uv pip install gradio protobuf

echo "Installing Qwen VL utilities..."
uv pip install qwen-vl-utils

echo "Installing image processing..."
uv pip install pillow

echo "Installing model management..."
uv pip install peft safetensors

echo "Installing config management..."
uv pip install pyyaml

echo "=========================================="""
echo "Verifying installation..."
echo "=========================================="

python - <<'VERIFY'
import sys
import torch
import diffusers
import transformers
import accelerate
import gradio
import PIL
import peft
import safetensors
import yaml

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"Diffusers: {diffusers.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Accelerate: {accelerate.__version__}")
print(f"Gradio: {gradio.__version__}")
print(f"PIL: {PIL.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"Safetensors: {safetensors.__version__}")
VERIFY

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source .venv/bin/activate"
echo "2. Create config.yaml with your model paths"
echo "3. Run the app:"
echo "   - GUI: python -m app.app"
echo "   - CLI: python -m src.cli --help"
echo ""
echo "Or use the convenience scripts:"
echo "   - ./app.sh    (start GUI)"
echo "   - ./go.sh     (quick start)"
echo ""
