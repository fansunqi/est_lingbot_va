#!/bin/bash
set -e

# ============================================================
# Entrypoint for lingbot-va Docker container
# Sets up lingbot-va uv environment on first run (cu124 variant)
# ============================================================

LINGBOT_DIR="/workspace/lingbot-va"
VENV_DIR="${LINGBOT_DIR}/.venv-docker"
MARKER="${VENV_DIR}/.docker-setup-done"

setup_lingbot_va() {
    echo "============================================"
    echo "Setting up lingbot-va environment (cu124)..."
    echo "============================================"

    cd "$LINGBOT_DIR"

    # Create a Docker-specific pyproject.toml adapted for CUDA 12.4 + Python 3.12
    cat > pyproject.docker.toml <<'PYPROJECT_EOF'
[project]
name = "LingBot_VA"
version = "0.0.0"
description = "LingBot-VA: A Pragmatic VA Foundation Model"
readme = "README.md"
requires-python = ">=3.12, <3.13"
dependencies = [
    "torch==2.9.0",
    "torchvision==0.24.0",
    "torchaudio==2.9.0",
    "lerobot>=0.5.0",
    "diffusers",
    "transformers",
    "accelerate",
    "einops",
    "easydict",
    "flash-attn",
    "numpy",
    "tqdm",
    "imageio[ffmpeg]",
    "websockets",
    "msgpack",
    "opencv-python",
    "matplotlib",
    "ftfy",
    "safetensors",
    "Pillow",
    "scipy",
    "wandb"
]

[[tool.uv.index]]
url = "https://download.pytorch.org/whl/cu124"
name = "pytorch-cu124"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu124" }
torchvision = { index = "pytorch-cu124" }
torchaudio = { index = "pytorch-cu124" }
flash-attn = { url = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu124torch2.9-cp312-cp312-linux_x86_64.whl" }
PYPROJECT_EOF

    # Use the Docker-specific pyproject.toml, install into a separate venv
    UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync --python 3.12 --pyproject pyproject.docker.toml

    touch "$MARKER"
    echo "============================================"
    echo "lingbot-va environment setup complete!"
    echo "============================================"
}

# Setup lingbot-va if not already done
if [ -d "$LINGBOT_DIR" ] && [ ! -f "$MARKER" ]; then
    setup_lingbot_va
fi

# Activate conda for interactive shells
eval "$(conda shell.bash hook)"

exec "$@"
