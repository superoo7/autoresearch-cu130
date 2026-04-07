#!/usr/bin/env bash
set -euo pipefail

echo "=== Autoresearch Setup ==="
echo "Target: RunPod RTX PRO 6000 Blackwell (96GB VRAM)"
echo "Template: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
echo ""

# Detect GPU
if command -v nvidia-smi &>/dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
fi
echo "Python: $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo ""

# 1. Install system tools
echo "Installing system tools..."
apt-get update -qq && apt-get install -y -qq tmux > /dev/null 2>&1
echo "tmux: $(tmux -V)"

# 2. Install uv
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# 3. Install Node.js + Claude Code CLI
if ! command -v node &>/dev/null; then
    echo "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y nodejs
fi
if ! command -v claude &>/dev/null; then
    echo "Installing Claude Code CLI..."
    npm install -g @anthropic-ai/claude-code
fi
echo "node: $(node --version)"

# 4. Create venv inheriting system PyTorch (already in RunPod template)
echo ""
echo "Creating Python venv (inheriting system packages)..."
PYTHON_BIN=$(which python3 || which python)
$PYTHON_BIN -m venv .venv --system-site-packages
echo "Venv created with system-site-packages (inherits PyTorch $(python -c 'import torch; print(torch.__version__)'))"

# 5. Install training deps (skip torch — already in system)
echo ""
echo "Installing Unsloth..."
.venv/bin/pip install --no-deps \
    "unsloth @ git+https://github.com/unslothai/unsloth" \
    "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo"

echo "Installing training dependencies..."
.venv/bin/pip install \
    "transformers>=5.2.0" "trl>=0.18.0" "peft>=0.15.0" "accelerate>=1.0.0" \
    datasets bitsandbytes Pillow huggingface_hub

# 6. Compile CUDA kernels for Blackwell (sm_120)
echo ""
echo "Compiling flash-linear-attention + causal-conv1d for Blackwell (sm_120)..."
export TORCH_CUDA_ARCH_LIST="12.0"
.venv/bin/pip install --no-build-isolation \
    "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention" \
    "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d"

# 7. Download datasets
echo ""
echo "Downloading datasets..."
.venv/bin/python prepare.py

# 8. Auto-configure train.py for this GPU (no thermal throttle needed)
echo ""
if grep -q 'THROTTLE_SECONDS = 1.0' train.py; then
    sed -i 's/THROTTLE_SECONDS = 1.0/THROTTLE_SECONDS = 0/' train.py
    echo "Set THROTTLE_SECONDS=0 (not needed on RTX PRO 6000)"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Login to HuggingFace:  huggingface-cli login"
echo "  2. Edit train.py CONFIG:"
echo "     - Set HF_UPLOAD = True"
echo "     - Set HF_REPO_ID = \"your-username/your-model\""
echo "  3. Run:  .venv/bin/python train.py 2>&1 | tee run.log"
echo ""
echo "After training, download GGUF on DGX Spark:"
echo "  huggingface-cli download <HF_REPO_ID> --local-dir ./model --include 'gguf/*'"
