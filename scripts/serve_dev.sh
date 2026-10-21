#!/usr/bin/env bash
set -euo pipefail

# Minimal launcher for development: foreground only, `<task>` as the first
# positional argument, every remaining argument is forwarded to hydra.
# For deployment (file checks, GPU selection, token file, background logs/PID,
# /readyz gating) use scripts/serve_deploy.sh instead.
#
# Usage:
#   bash scripts/serve_dev.sh <task> [hydra overrides...]
#
# Environment:
#   CKPT                 checkpoint path (required unless DRY_RUN=1)
#   DATASET_STATS        dataset_stats.json path (required unless DRY_RUN=1)
#   HOST / PORT          listen address (default 0.0.0.0:8765)
#   DRY_RUN=1            serve synthetic actions without loading weights
#   FASTWAM_SERVICE_TOKEN  enables token auth when set (recommended for wss)
#
# Examples:
#   CKPT=./runs/mydata/2026-01-01_00-00-00/checkpoints/weights/10000.pt \
#   DATASET_STATS=./runs/mydata/2026-01-01_00-00-00/dataset_stats.json \
#     bash scripts/serve_dev.sh mydata_uncond_2cam224_1e-4
#
#   DRY_RUN=1 bash scripts/serve_dev.sh libero_uncond_2cam224_1e-4

TASK="${1:?Usage: bash scripts/serve_dev.sh <task> [hydra overrides...]}"
shift

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
DRY_RUN="${DRY_RUN:-0}"
CKPT="${CKPT:-}"
DATASET_STATS="${DATASET_STATS:-}"

ARGS=(
  "task=${TASK}"
  "serving.host=${HOST}"
  "serving.port=${PORT}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=("serving.dry_run=true")
else
  if [[ -z "${CKPT}" || -z "${DATASET_STATS}" ]]; then
    echo "Error: set CKPT and DATASET_STATS (or DRY_RUN=1)." >&2
    exit 1
  fi
  ARGS+=(
    "serving.ckpt=${CKPT}"
    "serving.dataset_stats_path=${DATASET_STATS}"
  )
fi

# The package may be used straight from the checkout (no editable install).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "${REPO_ROOT}/src" ]]; then
  export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
fi

if [[ -n "${FASTWAM_SERVICE_TOKEN:-}" ]]; then
  echo "[serve] token auth enabled"
else
  echo "[serve] WARNING: FASTWAM_SERVICE_TOKEN is unset; the endpoint is open" >&2
fi

echo "[serve] task=${TASK} host=${HOST} port=${PORT} dry_run=${DRY_RUN}"
exec python "${REPO_ROOT}/scripts/serve.py" "${ARGS[@]}" "$@"
