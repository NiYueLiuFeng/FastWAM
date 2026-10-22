# FastWAM training / inference / LIBERO-eval image (x86_64).
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
#    - EGL/OpenGL stack: LIBERO OffScreenRenderEnv (MUJOCO_GL=egl)
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
# torch inductor/triton, wandb and LIBERO all write under it. Mounting a host
# directory over this path keeps that directory's own ownership instead.
RUN mkdir -p /workspace/home && chmod 777 /workspace/home

WORKDIR /workspace/FastWAM

# ---------------------------------------------------------------------------
# 5) FastWAM + LIBERO + sanity check
#    FastWAM code is baked into the image; data/checkpoints/runs are mounted.
#    The local version tags (+cu128) are stripped inside the image only, so the
#    installed cu126 wheels satisfy the pins. The repo file is left untouched.
#
#    LIBERO is the simulation environment for evaluation. Installed with
#    --no-deps so LIBERO's (older) requirements cannot downgrade fastwam's
#    pinned stack (numpy / transformers / etc.); its runtime deps are added
#    explicitly instead. mujoco is installed last to stay at 3.3.2 (must match
#    the LIBERO dataset version, per the README).
#
#    The layer ends with a build-time sanity check: dependency graph
#    consistent, key packages importable.
# ---------------------------------------------------------------------------
COPY . /workspace/FastWAM
RUN sed -i \
      -e 's/torch==2\.7\.1+cu128/torch==2.7.1/' \
      -e 's/torchvision==0\.22\.1+cu128/torchvision==0.22.1/' \
      /workspace/FastWAM/pyproject.toml \
  && grep -n 'torch==' /workspace/FastWAM/pyproject.toml \
  && pip install --no-cache-dir --retries 10 -e '/workspace/FastWAM[serving]'

# LIBERO's top-level `libero/` dir has no __init__.py (namespace package), so
# PEP 660 strict editable installs don't expose it: `import libero` fails unless
# the repo root is on sys.path. Put it on PYTHONPATH (also valid at runtime).
ENV PYTHONPATH=/opt/LIBERO

# LIBERO in its own layer so fastwam's install stays cached across rebuilds.
# Fetched as a codeload tarball instead of git clone: the container->github.com
# route is intermittent (curl 16 HTTP/2 framing errors, connect timeouts),
# while codeload.github.com serves reliably; the retry loop absorbs hiccups.
RUN for i in 1 2 3 4 5; do \
      wget -q --tries=3 --timeout=60 -O /tmp/libero.tar.gz \
        https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/refs/heads/master && break; \
      echo "libero download retry $i"; sleep $((i * 15)); \
    done && test -s /tmp/libero.tar.gz \
  && mkdir -p /opt/LIBERO \
  && tar xzf /tmp/libero.tar.gz -C /opt/LIBERO --strip-components=1 \
  && rm /tmp/libero.tar.gz \
  && pip install --no-cache-dir --retries 10 --no-deps -e /opt/LIBERO \
  && pip install --no-cache-dir \
    pyquaternion \
    easydict \
    pypng \
    pygame \
  && pip install --no-cache-dir --no-deps robosuite==1.4.1 \
  && pip install --no-cache-dir numba scipy pynput opencv-python \
  && pip install --no-cache-dir bddl==1.0.1 future pyparsing matplotlib cloudpickle "gym==0.25.2" \
  && pip install --no-cache-dir mujoco==3.3.2 \
  && mkdir -p /opt/LIBERO/libero/datasets /opt/libero-config /root/.libero \
  && python -c "import yaml; root='/opt/LIBERO/libero/libero'; yaml.dump({'benchmark_root': root, 'bddl_files': root+'/bddl_files', 'init_states': root+'/init_files', 'datasets': '/opt/LIBERO/libero/datasets', 'assets': root+'/assets'}, open('/opt/libero-config/config.yaml','w'))" \
  && cp /opt/libero-config/config.yaml /root/.libero/config.yaml \
  && chmod -R a+rX /opt/libero-config \
  && pip check \
  && python -c "import torch, torchvision, fastwam, deepspeed, av, torchcodec; print('torch', torch.__version__, 'cuda', torch.version.cuda)" \
  && python -c "import fastapi, uvicorn, msgpack, websockets; import fastwam.serving; print('serving ok')" \
  && python -c "from libero.libero import benchmark; from libero.libero.envs import OffScreenRenderEnv; print('libero ok')"

# WebSocket inference service port (see docs/serving_ws_api_zh.md).
EXPOSE 8765

# Seed ~/.libero/config.yaml into whatever $HOME the container starts with, so
# non-root runs (`--user <uid>:<gid>`) keep LIBERO working. Copied from the
# build context rather than symlinked into /workspace/FastWAM: that path is
# usually shadowed by a bind mount at runtime.
COPY scripts/container_entrypoint.sh /usr/local/bin/fastwam-entrypoint
RUN chmod 755 /usr/local/bin/fastwam-entrypoint

ENTRYPOINT ["/usr/local/bin/fastwam-entrypoint"]

# Default command: an interactive shell. See docker/run_container.sh for the
# recommended run invocation (GPU, shm, volume mounts).
CMD ["/bin/bash"]
