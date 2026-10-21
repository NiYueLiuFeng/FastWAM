#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# FastWAM 云端推理服务启动脚本（部署 / 运维用，变量驱动）
#
# 与 scripts/serve_dev.sh 的区别：本脚本会校验权重与统计文件是否存在、设置
# GPU、支持 token 文件、后台运行、写日志/PID，并等待 /readyz 就绪；
# 日常调试用 serve_dev.sh 更轻量。
#
# 最简用法（前台，Ctrl-C 停止）：
#   CKPT=/path/to/weights.pt \
#   DATASET_STATS=/path/to/dataset_stats.json \
#   TASK=mydata_uncond_2cam224_1e-4 \
#   GPU=0 PORT=8765 \
#     bash scripts/serve_deploy.sh
#
# 后台常驻 + token：
#   CKPT=... DATASET_STATS=... TASK=... BACKGROUND=1 \
#   TOKEN_FILE=/etc/fastwam/token bash scripts/serve_deploy.sh
#
# 无权重联调（不加载模型、不占显存）：
#   DRY_RUN=1 PORT=8765 bash scripts/serve_deploy.sh
#
# 变量说明（全部可用同名环境变量传入；命令行剩余参数会透传给 hydra）：
#   TASK                 任务配置名，对应 configs/task/<TASK>.yaml
#                        默认 libero_uncond_2cam224_1e-4
#   CKPT                 微调后的权重（.pt）；DRY_RUN=0 时必填
#   DATASET_STATS        与权重配套的 dataset_stats.json；DRY_RUN=0 时必填
#   GPU                  指定显卡编号，写入 CUDA_VISIBLE_DEVICES；空字符串表示不设置
#   HOST / PORT          监听地址，默认 0.0.0.0 / 8765
#   DTYPE                bf16 / fp16 / fp32，默认 bf16
#   TEXT_ENCODER_DEVICE  cuda（默认）或 cpu（显存不足时省 ~10GB）
#   STEPS                去噪步数，默认 10
#   SIGMA_SHIFT          null（默认，用 scheduler 默认）或浮点，旧版 release 权重常用 5.0
#   ACTION_HORIZON       null（默认 → num_frames-1）或整数
#   COMPILE              true（默认）/ false，是否 torch.compile 动作分支
#   MAX_QUEUE            推理队列长度，默认 16
#   REQUEST_TIMEOUT_MS   默认请求超时，默认 5000
#   TOKEN                访问 token；为空时读 TOKEN_FILE
#   TOKEN_FILE           token 文件，默认 /etc/fastwam/token（不存在则忽略、不启用鉴权）
#   DRY_RUN              1 = 不加载权重，返回合成动作，用于传输层联调
#   BACKGROUND           1 = nohup 后台运行（默认前台）
#   LOG_FILE             后台日志路径；默认 logs/serve_<TASK>_<时间>.log
#   WAIT_READY           1（默认，仅 BACKGROUND=1 生效）= 轮询 /readyz 后再退出
#   WAIT_TIMEOUT         等待就绪的最长秒数，默认 1800
#   PYTHON_BIN           解释器，默认 python（可指向 venv 里的 python）
#   EXTRA_ARGS           额外 hydra 覆盖（空格分隔），如 "serving.warmup_runs=3"
# ============================================================================

TASK="${TASK:-libero_uncond_2cam224_1e-4}"
CKPT="${CKPT:-}"
DATASET_STATS="${DATASET_STATS:-}"

GPU="${GPU-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"

DTYPE="${DTYPE:-bf16}"
TEXT_ENCODER_DEVICE="${TEXT_ENCODER_DEVICE:-cuda}"
STEPS="${STEPS:-10}"
SIGMA_SHIFT="${SIGMA_SHIFT:-null}"
ACTION_HORIZON="${ACTION_HORIZON:-null}"
COMPILE="${COMPILE:-true}"
MAX_QUEUE="${MAX_QUEUE:-16}"
REQUEST_TIMEOUT_MS="${REQUEST_TIMEOUT_MS:-5000}"

TOKEN="${TOKEN:-}"
TOKEN_FILE="${TOKEN_FILE:-/etc/fastwam/token}"
DRY_RUN="${DRY_RUN:-0}"

BACKGROUND="${BACKGROUND:-0}"
LOG_FILE="${LOG_FILE:-}"
WAIT_READY="${WAIT_READY:-1}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() {
  echo "Error: $*" >&2
  exit 1
}

is_true() {
  [[ "${1,,}" == "1" || "${1,,}" == "true" || "${1,,}" == "yes" ]]
}

is_false() {
  [[ "${1,,}" == "0" || "${1,,}" == "false" || "${1,,}" == "no" ]]
}

# ---------------------------------------------------------------- 校验输入
command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
  || die "找不到解释器 ${PYTHON_BIN}（可用 PYTHON_BIN=/path/to/venv/bin/python 指定）"

[[ -f "${REPO_ROOT}/configs/serve.yaml" ]] || die "仓库不完整：缺少 configs/serve.yaml"
[[ -f "${REPO_ROOT}/configs/task/${TASK}.yaml" ]] \
  || die "找不到任务配置 configs/task/${TASK}.yaml（TASK=${TASK}）"

if is_true "${DRY_RUN}"; then
  echo "[serve] DRY_RUN=1：不加载权重，返回合成动作"
else
  [[ -n "${CKPT}" ]] || die "CKPT 未设置（或设置 DRY_RUN=1 做无权重联调）"
  [[ -n "${DATASET_STATS}" ]] || die "DATASET_STATS 未设置（checkpoint 配套的 dataset_stats.json）"
  [[ -f "${CKPT}" ]] || die "CKPT 不存在：${CKPT}"
  [[ -f "${DATASET_STATS}" ]] || die "DATASET_STATS 不存在：${DATASET_STATS}"
fi

# ------------------------------------------------------------------ token
if [[ -z "${TOKEN}" && -n "${TOKEN_FILE}" && -f "${TOKEN_FILE}" ]]; then
  TOKEN="$(tr -d '[:space:]' < "${TOKEN_FILE}")"
fi
export FASTWAM_SERVICE_TOKEN="${TOKEN}"
if [[ -n "${TOKEN}" ]]; then
  echo "[serve] 已启用 token 鉴权"
else
  echo "[serve] WARNING: 未设置 token，接口对内网开放；生产环境请配置 FASTWAM_SERVICE_TOKEN" >&2
fi

# -------------------------------------------------------------------- GPU
if [[ -n "${GPU}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
  echo "[serve] CUDA_VISIBLE_DEVICES=${GPU}"
fi

# ------------------------------------------------------------- hydra 参数
ARGS=(
  "task=${TASK}"
  "serving.host=${HOST}"
  "serving.port=${PORT}"
  "serving.dtype=${DTYPE}"
  "serving.text_encoder_device=${TEXT_ENCODER_DEVICE}"
  "serving.num_inference_steps=${STEPS}"
  "serving.compile_action_infer=${COMPILE}"
  "serving.max_queue=${MAX_QUEUE}"
  "serving.request_timeout_ms=${REQUEST_TIMEOUT_MS}"
)

if is_true "${DRY_RUN}"; then
  ARGS+=("serving.dry_run=true")
else
  ARGS+=(
    "serving.dry_run=false"
    "serving.ckpt=${CKPT}"
    "serving.dataset_stats_path=${DATASET_STATS}"
  )
fi

if ! is_false "${SIGMA_SHIFT}" && [[ "${SIGMA_SHIFT}" != "null" ]]; then
  ARGS+=("serving.sigma_shift=${SIGMA_SHIFT}")
fi
if [[ "${ACTION_HORIZON}" != "null" && -n "${ACTION_HORIZON}" ]]; then
  ARGS+=("serving.action_horizon=${ACTION_HORIZON}")
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  read -r -a _extra <<<"${EXTRA_ARGS}"
  ARGS+=("${_extra[@]}")
fi
# 命令行剩余参数优先级最高，直接透传给 hydra。
ARGS+=("$@")

# ---------------------------------------------------------------- 启动
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[serve] task=${TASK} host=${HOST} port=${PORT} dry_run=${DRY_RUN}"
if ! is_true "${DRY_RUN}"; then
  echo "[serve] ckpt=${CKPT}"
  echo "[serve] stats=${DATASET_STATS}"
fi

print_usage_hint() {
  local query=""
  [[ -n "${TOKEN}" ]] && query="?token=<TOKEN>"
  echo "[serve] 健康检查: curl http://127.0.0.1:${PORT}/readyz"
  echo "[serve] 参考客户端: python examples/ws_client.py --url ws://127.0.0.1:${PORT}/v1/ws${query} --synthetic --repeat 5"
}

run_foreground() {
  echo "[serve] 前台运行，Ctrl-C 停止"
  print_usage_hint
  exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/serve.py" "${ARGS[@]}"
}

run_background() {
  if [[ -z "${LOG_FILE}" ]]; then
    local stamp
    stamp="$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="${REPO_ROOT}/logs/serve_${TASK}_${stamp}.log"
  fi
  mkdir -p "$(dirname "${LOG_FILE}")"

  nohup "${PYTHON_BIN}" "${REPO_ROOT}/scripts/serve.py" "${ARGS[@]}" \
    >"${LOG_FILE}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${LOG_FILE%.log}.pid"
  echo "[serve] 后台启动 pid=${pid} log=${LOG_FILE}"

  if is_true "${WAIT_READY}"; then
    echo "[serve] 等待 /readyz（模型加载 + warmup，最长 ${WAIT_TIMEOUT}s）..."
    local waited=0
    while (( waited < WAIT_TIMEOUT )); do
      if ! kill -0 "${pid}" 2>/dev/null; then
        echo "[serve] 进程已退出，最后 40 行日志：" >&2
        tail -n 40 "${LOG_FILE}" >&2 || true
        exit 1
      fi
      if curl -sf "http://127.0.0.1:${PORT}/readyz" >/dev/null 2>&1; then
        echo "[serve] 就绪（${waited}s）"
        print_usage_hint
        return 0
      fi
      sleep 2
      waited=$((waited + 2))
    done
    echo "[serve] WARNING: ${WAIT_TIMEOUT}s 内未就绪，进程仍在运行；查看 ${LOG_FILE}" >&2
    print_usage_hint
    return 0
  fi

  print_usage_hint
}

if is_true "${BACKGROUND}"; then
  run_background
else
  run_foreground
fi
