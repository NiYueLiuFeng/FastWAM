#!/usr/bin/env bash
# Launch an interactive FastWAM container with the standard volume layout.
#
# Usage:
#   bash docker/run_container.sh                     # uses default host dirs (repo-relative)
#   IMAGE=fastwam:cu126  HOST_DATA_DIR=/data bash docker/run_container.sh
#
# Optional env overrides:
#   IMAGE              image tag                      (default: fastwam:cu126)
#   HOST_DATA_DIR      datasets / text-embed cache    (default: ./data)
#   HOST_CKPT_DIR      Wan weights, backbone, ckpts   (default: ./checkpoints)
#   HOST_RUNS_DIR      training outputs               (default: ./runs)
#   HOST_EVAL_DIR      evaluation outputs             (default: ./evaluate_results)
#   HOST_HF_CACHE      huggingface download cache     (default: ./hf_cache; set empty to skip)
#   GPUS               GPUs to expose                 (default: all)
#   HOST_UID/HOST_GID  container runtime identity     (default: current host user)
#   HOST_USER          USER/LOGNAME inside container  (default: current host user)
#   HOST_HOME_DIR      host dir mounted as $HOME      (default: ./.container_home)
#   EXTRA_DOCKER_ARGS  appended verbatim to docker run
#
# The container runs as the host uid/gid, so everything written through the
# bind mounts stays host-owned. Files created earlier by a root container must
# be fixed once (see the chown snippet in the README/answer).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
HOST_USER="${HOST_USER:-$(id -un)}"
HOST_HOME_DIR="${HOST_HOME_DIR:-${REPO_ROOT}/.container_home}"

NAME=${NAME:-""}

IMAGE="${IMAGE:-fastwam:cu126}"
GPUS="${GPUS:-all}"
HOST_DATA_DIR="${HOST_DATA_DIR:-${REPO_ROOT}/data}"
HOST_CKPT_DIR="${HOST_CKPT_DIR:-${REPO_ROOT}/checkpoints}"
HOST_RUNS_DIR="${HOST_RUNS_DIR:-${REPO_ROOT}/runs}"
HOST_EVAL_DIR="${HOST_EVAL_DIR:-${REPO_ROOT}/evaluate_results}"
HOST_HF_CACHE="${HOST_HF_CACHE:-${REPO_ROOT}/hf_cache}"

for d in "${HOST_DATA_DIR}" "${HOST_CKPT_DIR}" "${HOST_RUNS_DIR}" "${HOST_EVAL_DIR}" "${HOST_HOME_DIR}"; do
  mkdir -p "${d}"
done

VOLUME_ARGS=(
  -v "${HOST_DATA_DIR}:/workspace/data"
  -v "${HOST_CKPT_DIR}:/workspace/checkpoints"
  -v "${HOST_RUNS_DIR}:/workspace/FastWAM/runs"
  -v "${HOST_EVAL_DIR}:/workspace/FastWAM/evaluate_results"
)
if [[ -n "${HOST_HF_CACHE}" ]]; then
  mkdir -p "${HOST_HF_CACHE}"
  VOLUME_ARGS+=(-v "${HOST_HF_CACHE}:/workspace/hf_cache")
fi

# $HOME must be writable: torch inductor/triton, wandb and LIBERO all write
# under it, and the image has no passwd entry for the host uid.
VOLUME_ARGS+=(-v "${HOST_HOME_DIR}:/workspace/home")

USER_ARGS=(
  --user "${HOST_UID}:${HOST_GID}"
  -e HOME=/workspace/home
  -e USER="${HOST_USER}"
  -e LOGNAME="${HOST_USER}"
  -e XDG_CACHE_HOME=/workspace/home/.cache
  -e TRITON_CACHE_DIR=/workspace/home/.triton
)

exec docker run -it \
  --gpus "${GPUS}" \
  --ipc=host \
  --shm-size=32g \
  --network=host \
  --name=fastwam-"${NAME}"  \
  "${USER_ARGS[@]}" \
  "${VOLUME_ARGS[@]}" \
  ${EXTRA_DOCKER_ARGS:-} \
  "${IMAGE}" "${@:-/bin/bash}"
