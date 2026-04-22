# ============================================================
# Docker image for lingbot-va evaluation
# Targets: NVIDIA driver 550 + CUDA 12.4 + Vulkan + RTX 4090
# Contains: RoboTwin (conda, Python 3.10) + lingbot-va (uv, Python 3.12)
# ============================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ============ System dependencies ============
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Vulkan (for SAPIEN rendering)
    libvulkan1 libvulkan-dev vulkan-tools mesa-vulkan-drivers \
    # OpenGL / rendering
    libgl1-mesa-glx libgl1-mesa-dev libglib2.0-0 \
    libegl1-mesa-dev libgles2-mesa-dev \
    libsm6 libxext6 libxrender1 libxkbcommon0 \
    # Media
    ffmpeg \
    # Build tools
    git wget curl build-essential cmake \
    # Utilities
    tmux htop \
    && rm -rf /var/lib/apt/lists/*

# ============ Miniconda (for RoboTwin, Python 3.10) ============
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/miniconda/bin:$PATH"

# Create RoboTwin conda environment
RUN conda create -n RoboTwin python=3.10 -y

# Install PyTorch for RoboTwin (cu121, compatible with driver 550)
RUN conda run --no-capture-output -n RoboTwin \
    pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install pytorch3d (compile from source, needs torch first)
RUN conda run --no-capture-output -n RoboTwin \
    pip install --no-cache-dir \
    "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8"

# Install remaining RoboTwin dependencies
COPY requirements-robotwin.txt /tmp/requirements-robotwin.txt
RUN conda run --no-capture-output -n RoboTwin \
    pip install --no-cache-dir -r /tmp/requirements-robotwin.txt

# ============ uv (for lingbot-va, Python 3.12) ============
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Pre-install Python 3.12 for uv
RUN uv python install 3.12

# ============ Environment ============
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

# Bypass proxy for localhost websocket connections
ENV no_proxy=localhost,127.0.0.1,0.0.0.0
ENV NO_PROXY=localhost,127.0.0.1,0.0.0.0

WORKDIR /workspace

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
