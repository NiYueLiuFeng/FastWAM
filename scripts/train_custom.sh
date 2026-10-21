#!/usr/bin/env bash
# One-command training launcher for a custom LeRobot dataset.
#
#   bash scripts/train_custom.sh --dataset ./data/my_dataset_v30 --gpus 4
#   bash scripts/train_custom.sh --task my_task --gpus 8
#   bash scripts/train_custom.sh --task-file /path/to/my_task.yaml --gpus 8
#   bash scripts/train_custom.sh --data-config /path/to/my_data.yaml --gpus 8
#
# The dataset mode reads meta/info.json (LeRobot 2.0/2.1/3.0), generates
#   configs/data/autogen_<name>.yaml
#   configs/task/autogen_<name>.yaml
# then checks the ActionDiT backbone and text-embedding cache. Missing
# prerequisites are reported together with the exact command to run; nothing
# heavy is executed automatically.
#
# Everything after `--` is forwarded verbatim as Hydra overrides.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIG_PWD="$(pwd)"
PYTHON="${PYTHON:-python}"
PREPARE="${REPO_ROOT}/scripts/prepare_custom_training.py"

DATASETS=()
GEN_EXTRA=()
EXTRA=()
PASSTHRU=()

TASK=""
TASK_FILE=""
DATA_CONFIG=""
NAME=""
GPUS=""
MODEL=""
EPOCHS=""
BATCH_SIZE=""
LR=""
NUM_WORKERS=""
SAVE_EVERY=""
RESUME=""
NORM_STATS=""
BACKBONE=""
ZERO=1
ONLINE_TEXT=0
RANDOM_BACKBONE=0
SKIP_PREREQ=0
DRY_RUN=0
FORCE=0

usage() {
  cat <<'EOF'
用法:
  bash scripts/train_custom.sh --dataset PATH [--dataset PATH ...] [选项]
  bash scripts/train_custom.sh --task NAME [选项]
  bash scripts/train_custom.sh --task-file /path/to/task.yaml [选项]
  bash scripts/train_custom.sh --data-config /path/to/data.yaml [选项]

数据集模式 (--dataset):
  --name NAME              配置名（默认取数据集目录名）
  --camera KEY             指定相机（去掉 observation.images. 前缀，可重复）
  --exclude-camera KEY     排除相机（可重复）
  --intersect-cameras      只保留所有数据集共有的相机
  --resize H W             每个相机 resize 尺寸（默认 224 224）
  --concat LAYOUT          horizontal / vertical / robotwin / null（默认 auto）
  --video-size H W         拼接后的最终尺寸（H、W 必须是 16 的倍数）
  --num-frames N           默认 33
  --action-video-freq-ratio N  默认 4
  --global-sample-stride N     默认 1
  --action-key KEY         action.<key> 子特征（可重复）
  --state-key KEY          observation.state.<key> 子特征（可重复）
  --delta-mask SPEC        true/false 逗号列表，或 all-false / all-true
  --norm-mode MODE         min/max（默认）或 z-score
  --val-proportion P       留出验证集比例（默认 0）
  --context-len N          文本编码长度（默认 128）
  --cache-dir PATH         文本嵌入缓存目录

训练选项:
  --gpus N                 使用卡数（默认：当前可见 GPU 数）
  --model NAME             fastwam / fastwam_idm / fastwam_joint / fastwam_optional_idm
  --epochs N, --batch-size N, --lr LR, --num-workers N, --save-every N
  --resume PATH            断点续训（目录或权重 .pt）
  --norm-stats PATH        data.train.pretrained_norm_stats
  --backbone PATH          自定义 ActionDiT backbone 路径
  --random-backbone-init   跳过 backbone 检查并随机初始化（不推荐）
  --online-text-encode     不预计算文本嵌入，训练时在线编码
  --zero {1,2}             DeepSpeed ZeRO stage（默认 1）
  --skip-prereq-check      只做配置校验，不检查 backbone / 文本缓存
  --dry-run                打印最终训练命令后退出
  --force                  允许覆盖非自动生成的同名配置
  --extra "k=v k=v"        追加 Hydra overrides（可用于覆盖上面的默认值）
  --                       其后的参数原样传给训练脚本
  -h, --help               显示本帮助
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

resolve_path() {
  realpath -m -- "$1"
}

detect_gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local IFS=','
    # shellcheck disable=SC2206
    local devices=(${CUDA_VISIBLE_DEVICES})
    echo "${#devices[@]}"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi -L | wc -l | tr -d ' '
  else
    echo 0
  fi
}

need_value() {
  [[ $# -ge 2 ]] || die "$1 需要一个参数"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      need_value "$@"
      DATASETS+=("$(resolve_path "$2")")
      shift 2
      ;;
    --task)
      need_value "$@"
      TASK="$2"
      shift 2
      ;;
    --task-file)
      need_value "$@"
      TASK_FILE="$(resolve_path "$2")"
      shift 2
      ;;
    --data-config)
      need_value "$@"
      if [[ -f "$2" ]]; then
        DATA_CONFIG="$(resolve_path "$2")"
      else
        DATA_CONFIG="$2"
      fi
      shift 2
      ;;
    --name)
      need_value "$@"
      NAME="$2"
      shift 2
      ;;
    --gpus)
      need_value "$@"
      GPUS="$2"
      shift 2
      ;;
    --model)
      need_value "$@"
      MODEL="$2"
      shift 2
      ;;
    --epochs)
      need_value "$@"
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      need_value "$@"
      BATCH_SIZE="$2"
      shift 2
      ;;
    --lr)
      need_value "$@"
      LR="$2"
      shift 2
      ;;
    --num-workers)
      need_value "$@"
      NUM_WORKERS="$2"
      shift 2
      ;;
    --save-every)
      need_value "$@"
      SAVE_EVERY="$2"
      shift 2
      ;;
    --resume)
      need_value "$@"
      RESUME="$(resolve_path "$2")"
      shift 2
      ;;
    --norm-stats)
      need_value "$@"
      NORM_STATS="$(resolve_path "$2")"
      shift 2
      ;;
    --backbone)
      need_value "$@"
      BACKBONE="$(resolve_path "$2")"
      shift 2
      ;;
    --online-text-encode) ONLINE_TEXT=1; shift ;;
    --random-backbone-init) RANDOM_BACKBONE=1; shift ;;
    --skip-prereq-check) SKIP_PREREQ=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    --zero)
      need_value "$@"
      [[ "$2" == "1" || "$2" == "2" ]] || die "--zero 只支持 1 或 2"
      ZERO="$2"
      shift 2
      ;;
    --camera|--exclude-camera|--concat|--norm-mode|--delta-mask|--cache-dir)
      need_value "$@"
      GEN_EXTRA+=("$1" "$2")
      shift 2
      ;;
    --resize|--video-size)
      [[ $# -ge 3 ]] || die "$1 需要两个参数 (H W)"
      GEN_EXTRA+=("$1" "$2" "$3")
      shift 3
      ;;
    --num-frames|--action-video-freq-ratio|--global-sample-stride|--val-proportion|--context-len)
      need_value "$@"
      GEN_EXTRA+=("$1" "$2")
      shift 2
      ;;
    --action-key|--state-key)
      need_value "$@"
      GEN_EXTRA+=("$1" "$2")
      shift 2
      ;;
    --intersect-cameras)
      GEN_EXTRA+=("--intersect-cameras")
      shift
      ;;
    --extra)
      need_value "$@"
      read -r -a _extra <<< "$2"
      EXTRA+=("${_extra[@]}")
      shift 2
      ;;
    --)
      shift
      PASSTHRU+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "未知参数: $1（用 --help 查看用法）"
      ;;
  esac
done

mode_count=0
if [[ ${#DATASETS[@]} -gt 0 ]]; then mode_count=$((mode_count + 1)); fi
if [[ -n "$TASK" ]]; then mode_count=$((mode_count + 1)); fi
if [[ -n "$TASK_FILE" ]]; then mode_count=$((mode_count + 1)); fi
if [[ -n "$DATA_CONFIG" ]]; then mode_count=$((mode_count + 1)); fi
if [[ $mode_count -ne 1 ]]; then
  die "必须且只能提供 --dataset / --task / --task-file / --data-config 之一"
fi

if [[ -z "$GPUS" ]]; then
  GPUS="$(detect_gpu_count)"
  if [[ "$GPUS" == "0" ]]; then
    die "未检测到可见 GPU；请用 --gpus N 显式指定（或检查驱动/容器 --gpus 参数）"
  fi
fi
[[ "$GPUS" =~ ^[0-9]+$ ]] && (( GPUS >= 1 )) || die "--gpus 必须是正整数，实际为 $GPUS"

visible="$(detect_gpu_count)"
if [[ "$visible" != "0" ]] && (( GPUS > visible )); then
  die "--gpus=$GPUS 超过当前可见 GPU 数 ($visible)"
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  die "找不到 Python: $PYTHON"
fi

cd "$REPO_ROOT"

for cmd in generate import-config override-form verify; do
  "$PYTHON" "$PREPARE" "$cmd" --help >/dev/null 2>&1 \
    || die "无法执行 $PREPARE $cmd；请检查 PYTHON=$PYTHON 与仓库完整性"
done

TASK_NAME=""
if [[ ${#DATASETS[@]} -gt 0 ]]; then
  GEN_CMD=("$PYTHON" "$PREPARE" generate)
  for dataset in "${DATASETS[@]}"; do
    GEN_CMD+=(--dataset "$dataset")
  done
  if [[ -n "$NAME" ]]; then GEN_CMD+=(--name "$NAME"); fi
  if [[ -n "$MODEL" ]]; then GEN_CMD+=(--model "$MODEL"); fi
  if (( FORCE )); then GEN_CMD+=(--force); fi
  GEN_CMD+=("${GEN_EXTRA[@]}" --print-name)
  TASK_NAME="$("${GEN_CMD[@]}")"
elif [[ -n "$TASK" && ( "$TASK" == */* || "$TASK" == *.yaml ) ]]; then
  IMPORT_CMD=("$PYTHON" "$PREPARE" import-config --task-file "$(resolve_path "$TASK")")
  if [[ -n "$MODEL" ]]; then IMPORT_CMD+=(--model "$MODEL"); fi
  if (( FORCE )); then IMPORT_CMD+=(--force); fi
  IMPORT_CMD+=(--print-name)
  TASK_NAME="$("${IMPORT_CMD[@]}")"
elif [[ -n "$TASK" ]]; then
  [[ -f "$REPO_ROOT/configs/task/${TASK}.yaml" ]] \
    || die "找不到 task 配置: configs/task/${TASK}.yaml（或用 --task-file 指定路径）"
  TASK_NAME="$TASK"
elif [[ -n "$TASK_FILE" ]]; then
  IMPORT_CMD=("$PYTHON" "$PREPARE" import-config --task-file "$TASK_FILE")
  if [[ -n "$MODEL" ]]; then IMPORT_CMD+=(--model "$MODEL"); fi
  if (( FORCE )); then IMPORT_CMD+=(--force); fi
  IMPORT_CMD+=(--print-name)
  TASK_NAME="$("${IMPORT_CMD[@]}")"
else
  IMPORT_CMD=("$PYTHON" "$PREPARE" import-config --data-config "$DATA_CONFIG")
  if [[ -n "$MODEL" ]]; then IMPORT_CMD+=(--model "$MODEL"); fi
  if (( FORCE )); then IMPORT_CMD+=(--force); fi
  IMPORT_CMD+=(--print-name)
  TASK_NAME="$("${IMPORT_CMD[@]}")"
fi

[[ -n "$TASK_NAME" ]] || die "生成/导入配置失败，未得到 task 名称"

SETS=()
if [[ -n "$MODEL" ]]; then SETS+=("model=$MODEL"); fi
if [[ -n "$EPOCHS" ]]; then SETS+=("num_epochs=$EPOCHS"); fi
if [[ -n "$BATCH_SIZE" ]]; then SETS+=("batch_size=$BATCH_SIZE"); fi
if [[ -n "$LR" ]]; then SETS+=("learning_rate=$LR"); fi
if [[ -n "$NUM_WORKERS" ]]; then SETS+=("num_workers=$NUM_WORKERS"); fi
if [[ -n "$SAVE_EVERY" ]]; then SETS+=("save_every=$SAVE_EVERY"); fi
if [[ -n "$RESUME" ]]; then SETS+=("resume=$RESUME"); fi
if [[ -n "$NORM_STATS" ]]; then SETS+=("data.train.pretrained_norm_stats=$NORM_STATS"); fi
if [[ -n "$BACKBONE" ]]; then SETS+=("model.action_dit_pretrained_path=$BACKBONE"); fi
if (( RANDOM_BACKBONE )); then
  SETS+=("model.action_dit_pretrained_path=null" "model.skip_dit_load_from_pretrain=true")
fi
if (( ONLINE_TEXT )); then
  SETS+=("data.train.use_text_embed_cache=false" "model.load_text_encoder=true")
fi

NORMALIZED=()
if [[ ${#SETS[@]} -gt 0 ]]; then
  SET_ARGS=()
  for item in "${SETS[@]}"; do SET_ARGS+=(--set "$item"); done
  NORMALIZED_OUT="$("$PYTHON" "$PREPARE" override-form --task "$TASK_NAME" "${SET_ARGS[@]}")"
  if [[ -n "$NORMALIZED_OUT" ]]; then
    mapfile -t NORMALIZED <<< "$NORMALIZED_OUT"
  fi
fi

OVERRIDES=("${NORMALIZED[@]}" "${EXTRA[@]}" "${PASSTHRU[@]}")

VERIFY_CMD=("$PYTHON" "$PREPARE" verify --task "$TASK_NAME" --gpus "$GPUS")
for item in "${OVERRIDES[@]}"; do VERIFY_CMD+=(--override "$item"); done
if (( ONLINE_TEXT )); then VERIFY_CMD+=(--online-text-encode); fi
if (( RANDOM_BACKBONE )); then VERIFY_CMD+=(--random-backbone-init); fi
if (( SKIP_PREREQ )); then VERIFY_CMD+=(--skip-prereq-check); fi

status=0
"${VERIFY_CMD[@]}" || status=$?
if (( status != 0 )); then
  exit "$status"
fi

LAUNCH=("bash" "scripts/train_zero${ZERO}.sh" "$GPUS" "task=$TASK_NAME" "${OVERRIDES[@]}")

if (( DRY_RUN )); then
  echo "[dry-run] cd $REPO_ROOT"
  printf '[dry-run]'
  printf ' %q' "${LAUNCH[@]}"
  printf '\n'
  exit 0
fi

echo "[train_custom] task=$TASK_NAME gpus=$GPUS zero=$ZERO" >&2
exec "${LAUNCH[@]}"
