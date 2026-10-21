#!/usr/bin/env bash
# Launch an interactive FastWAM container with the standard volume layout.
#
# Usage:
#   bash docker/run_container.sh                     # uses default host dirs (repo-relative)
#   IMAGE=fastwam:cu126 HOST_DATA_DIR=/data bash docker/run_container.sh
#
# Optional env overrides:
#   IMAGE              image tag                      (default: fastwam:cu126)
#   HOST_DATA_DIR      datasets / text-embed cache    (default: ./data)
#   HOST_CKPT_DIR      Wan weights, backbone, ckpts   (default: ./checkpoints)
#   HOST_RUNS_DIR      training outputs               (default: ./runs)
#   HOST_EVAL_DIR      evaluation outputs             (default: ./evaluate_results)
#   HOST_HF_CACHE      huggingface download cache     (default: ./hf_cache; set empty to skip)
#   GPUS               GPUs to expose                 (default: all)
#   EXTRA_DOCKER_ARGS  appended verbatim to docker run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${IMAGE:-fastwam:cu126}"
GPUS="${GPUS:-all}"
HOST_DATA_DIR="${HOST_DATA_DIR:-${REPO_ROOT}/data}"
HOST_CKPT_DIR="${HOST_CKPT_DIR:-${REPO_ROOT}/checkpoints}"
HOST_RUNS_DIR="${HOST_RUNS_DIR:-${REPO_ROOT}/runs}"
HOST_EVAL_DIR="${HOST_EVAL_DIR:-${REPO_ROOT}/evaluate_results}"
HOST_HF_CACHE="${HOST_HF_CACHE:-${REPO_ROOT}/hf_cache}"

for d in "${HOST_DATA_DIR}" "${HOST_CKPT_DIR}" "${HOST_RUNS_DIR}" "${HOST_EVAL_DIR}"; do
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

exec docker run --rm -it \
  --gpus "${GPUS}" \
  --ipc=host \
  --shm-size=32g \
  --network=host \
  "${VOLUME_ARGS[@]}" \
  ${EXTRA_DOCKER_ARGS:-} \
  "${IMAGE}" "${@:-/bin/bash}"
