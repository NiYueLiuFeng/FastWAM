# FastWAM training / inference image (x86_64). LIBERO is NOT installed: mount
# a LIBERO checkout at runtime if you need the simulation evaluation env.
#
# Host driver requirement: any CUDA 12.x driver (>= 525, e.g. a 12.1-era ~530 driver).
# The torch cu126 wheels bundle their own CUDA runtime and work on all 12.x drivers
# via CUDA minor version compatibility.
#
# GPU arch: the cu126 wheels ship kernels for sm_50..sm_90 only. Blackwell
# consumer cards (RTX 50-series, sm_120) fail with "no kernel image is available";
# for those, install torch==2.7.1+cu128 / torchvision==0.22.1+cu128 instead.
#
# The image also serves WebSocket inference:
#   python scripts/serve.py task=<task> serving.ckpt=... serving.dataset_stats_path=...
# See docs/serving_ws_api_zh.md.
#
# Build (from the repo root):
#   docker build -f docker/Dockerfile -t fastwam:cu126 .
#
# If Docker Hub is unreachable from the daemon, point CUDA_IMAGE at a registry
# mirror, e.g.:
#   docker build -f docker/Dockerfile -t fastwam:cu126 \
#     --build-arg CUDA_IMAGE=docker.m.daocloud.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 .
#
# Base is the *devel* image so nvcc is present for DeepSpeed JIT op compilation.

ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
FROM --platform=linux/amd64 ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Same story for pip: files.pythonhosted.org crawls from CN networks (~KB/s).
# Route all pip installs (torch, fastwam, LIBERO deps) through the Huawei
# Cloud PyPI mirror (~10MB/s). The PyPI build of torch 2.7.1 IS the cu126
# variant (torch.version.cuda == "12.6"), so no special index is needed.
ENV PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
    PIP_DEFAULT_TIMEOUT=60

# The base image points apt at archive.ubuntu.com, which crawls in CN networks.
# Swap to the Huawei Cloud Ubuntu mirror (same one the build host uses).
RUN sed -i \
      -e 's|archive.ubuntu.com|mirrors.huaweicloud.com|g' \
      -e 's|security.ubuntu.com|mirrors.huaweicloud.com|g' \
      /etc/apt/sources.list

# ---------------------------------------------------------------------------
# 1) System packages + Python 3.12
#    - ffmpeg: video decoding (torchcodec requires system FFmpeg libs)
#    - EGL/OpenGL stack: offscreen rendering (MUJOCO_GL=egl), used by a
#      LIBERO checkout mounted at runtime rather than installed here
#    - build-essential / ninja: DeepSpeed JIT compilation
#    - Python 3.12 via deadsnakes (pyproject requires >=3.12; Ubuntu 22.04
#      ships 3.10)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    ninja-build \
    git \
    ca-certificates \
    wget \
    ffmpeg \
    libgl1 \
    libegl1 \
    libgles2 \
    libglvnd0 \
    libglx0 \
  && add-apt-repository -y ppa:deadsnakes/ppa \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
  && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2) venv + PyTorch cu126 (no cu121 build exists for torch 2.7.1; cu126 runs
#    on any CUDA 12.x driver). Install before fastwam so the pinned deps are
#    satisfied. Kept in its own layer (before COPY) so the heavy torch wheels
#    stay cached across source-code changes. torch 2.7.1 from PyPI is the
#    cu126 build; --retries absorbs mirror hiccups.
# ---------------------------------------------------------------------------
RUN python3.12 -m venv /opt/venv \
  && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
  && /opt/venv/bin/pip install --no-cache-dir --retries 10 \
    torch==2.7.1 \
    torchvision==0.22.1

ENV PATH=/opt/venv/bin:${PATH}

# ---------------------------------------------------------------------------
# 4) Runtime environment
# ---------------------------------------------------------------------------
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MUJOCO_GL=egl \
    DIFFSYNTH_MODEL_BASE_PATH=/workspace/checkpoints \
    HF_HOME=/workspace/hf_cache \
    HOME=/workspace/home \
    XDG_CACHE_HOME=/workspace/home/.cache \
    TRITON_CACHE_DIR=/workspace/home/.triton \
    TORCHINDUCTOR_CACHE_DIR=/workspace/home/.torchinductor

# Writable $HOME so the image also works with `docker run --user $(id -u):$(id -g)`:
# torch inductor/triton and wandb all write under it. Mounting a host
# directory over this path keeps that directory's own ownership instead.
RUN mkdir -p /workspace/home && chmod 777 /workspace/home

WORKDIR /workspace/FastWAM

# ---------------------------------------------------------------------------
# 5) FastWAM + sanity check
#    FastWAM code is baked into the image; data/checkpoints/runs are mounted.
#    The local version tags (+cu128) are stripped inside the image only, so the
#    installed cu126 wheels satisfy the pins. The repo file is left untouched.
#
#    LIBERO is deliberately not installed (robosuite/mujoco/bddl etc. are
#    LIBERO-only deps). Mount a LIBERO checkout at runtime if you need it.
#
#    The layer ends with a build-time sanity check: dependency graph
#    consistent, key packages importable, and uvicorn actually able to speak
#    WebSocket. That last one matters: uvicorn picks its WS backend at import
#    time and, with no backend, degrades silently -- upgrade requests are
#    handled as plain GETs, so /v1/ws answers 404 and clients die with
#    InvalidStatus. A bare `pip install uvicorn` (no [standard] extra) is
#    exactly that state; assert it here instead of debugging it at runtime.
#    The `[serving]` extra covers fastapi/uvicorn[standard]/msgpack/websockets.
# ---------------------------------------------------------------------------
COPY . /workspace/FastWAM
RUN sed -i \
      -e 's/torch==2\.7\.1+cu128/torch==2.7.1/' \
      -e 's/torchvision==0\.22\.1+cu128/torchvision==0.22.1/' \
      /workspace/FastWAM/pyproject.toml \
  && grep -n 'torch==' /workspace/FastWAM/pyproject.toml \
  && pip install --no-cache-dir --retries 10 -e '/workspace/FastWAM[serving]' \
  && pip check \
  && python -c "import torch, torchvision, fastwam, deepspeed, av, torchcodec; print('torch', torch.__version__, 'cuda', torch.version.cuda)" \
  && python -c "import fastapi, uvicorn, msgpack, websockets; import fastwam.serving; print('serving ok', 'fastapi', fastapi.__version__, 'uvicorn', uvicorn.__version__, 'websockets', websockets.__version__)" \
  && python -c "from uvicorn.protocols.websockets.auto import AutoWebSocketsProtocol as ws; assert ws is not None, 'uvicorn has no WebSocket backend: /v1/ws would answer HTTP 404'; print('ws backend', ws.__name__)"

# WebSocket inference service port (see docs/serving_ws_api_zh.md).
EXPOSE 8765

# Default command: an interactive shell. See docker/run_container.sh for the
# recommended run invocation (GPU, shm, volume mounts).
CMD ["/bin/bash"]
