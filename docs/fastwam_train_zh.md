# FastWAM 自定义数据一键训练：实现方案与验收总结

本文记录 `scripts/train_custom.sh` 与 `scripts/prepare_custom_training.py` 的设计与实现结果。
面向的使用场景是：**手里有一份 LeRobot 2.0 / 2.1 / 3.0 格式的数据，传数据集路径就能开始训练。**

面向使用者的分步教程见 [custom_data_tutorial_zh.md](custom_data_tutorial_zh.md)；
容器与多卡启动见 [fastwam_docker.md](fastwam_docker.md)。

---

## 1. 背景

仓库原有的训练入口是：

```bash
bash scripts/train_zero1.sh <nproc_per_node> [hydra overrides...]
# 例：bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4
```

它**没有数据集路径参数**，只认已经写好的 `task=` 配置。虽然可以用
`data.train.dataset_dirs=[/abs/path]` 临时覆盖路径，但新数据集的相机名、
action/state 维度、`video_size`、`num_output_cameras` 等仍必须与所选 data config
一致，否则会在建 dataloader 或模型输出层时报错。也就是说，新数据集必须先按
`configs/data/*.yaml` + `configs/task/*.yaml` 手写两份配置。

本方案新增一层「配置生成 + 校验」的封装，复用现有训练入口，不重复实现分布式启动逻辑。
两个已确认的设计决策：

1. **自动探测数据集**：读取 LeRobot `meta/info.json` 推导 reader 版本、相机、维度，
   生成配置；同时支持直接传已有配置。
2. **前置步骤只检查、不代跑**：ActionDiT backbone 与文本嵌入缓存缺失时，
   打印准确命令并以退出码 `2` 结束，不自动执行重任务。

---

## 2. 方案概览

| 文件 | 类型 | 作用 |
| --- | --- | --- |
| `scripts/prepare_custom_training.py` | 新增 | 探测数据集、生成/导入配置、Hydra 合成校验、前置条件检查 |
| `scripts/train_custom.sh` | 新增 | 参数解析 → 生成/导入 → 校验 → `exec train_zero{1,2}.sh` |
| `scripts/precompute_text_embeds.py` | 修改 | 读取任务文本时增加 LeRobot 3.0 `meta/tasks.parquet` 回退 |
| `docs/custom_data_tutorial_zh.md` | 修改 | 顶部新增「一键启动（推荐）」小节 |
| `docs/fastwam_train_zh.md` | 新增 | 本文：实现方案与验收总结 |

调用链：

```text
bash scripts/train_custom.sh --dataset <ds> [options]
        │
        ├─ prepare_custom_training.py generate        # 探测 + 写配置文件
        │     → configs/data/autogen_<slug>.yaml
        │     → configs/task/autogen_<slug>.yaml
        ├─ prepare_custom_training.py override-form   # 补齐 Hydra 的 + 前缀
        ├─ prepare_custom_training.py verify          # 语义 + 前置条件
        │     退出码 0 通过 / 1 配置错误 / 2 前置条件缺失
        └─ exec bash scripts/train_zero1.sh <gpus> task=autogen_<slug> <overrides>
```

最终仍由 `train_zero1.sh` / `train_zero2.sh` 负责 `accelerate launch`、`RUN_ID`、
多机环境变量和 DeepSpeed 配置，输出目录继续是 `runs/<task>/<run_id>/`。

---

## 3. 使用方式

### 3.1 四种入口（互斥，必须选一个）

```bash
# 1) 数据集路径：自动探测并生成配置
bash scripts/train_custom.sh --dataset ./data/my_dataset_v30 --gpus 4

# 2) 多个数据集（reader 版本、fps、相机集合、action/state 维度必须一致）
bash scripts/train_custom.sh \
  --dataset ./data/ds_a --dataset ./data/ds_b --gpus 4

# 3) 已有 task 配置（configs/task/<name>.yaml，或直接给 .yaml 路径）
bash scripts/train_custom.sh --task my_task --gpus 4
bash scripts/train_custom.sh --task /path/to/my_task.yaml --gpus 4

# 4) 已有 data 配置（仓库内名称或仓库外路径，会自动生成 task 包装）
bash scripts/train_custom.sh --data-config /path/to/my_data.yaml --gpus 4
```

`--task` 的值如果包含 `/` 或以 `.yaml` 结尾，会自动按 `--task-file` 处理。

不传 `--gpus` 时，按 `CUDA_VISIBLE_DEVICES` 的条目数或 `nvidia-smi -L` 的行数推断；
推断结果为 0 时直接报错，提示显式指定。`--gpus` 大于当前可见卡数也会报错。

### 3.2 数据集策略参数

| 参数 | 说明 |
| --- | --- |
| `--name NAME` | 配置名，默认取数据集目录名；多数据集时为 `<第一个目录名>_and_<N-1>` |
| `--camera KEY` | 指定相机（去掉 `observation.images.` 前缀，可重复，决定顺序） |
| `--exclude-camera KEY` | 排除相机（可重复） |
| `--intersect-cameras` | 多数据集只保留公共相机 |
| `--resize H W` | 每相机 resize 尺寸，默认 `224 224` |
| `--concat LAYOUT` | `horizontal` / `vertical` / `robotwin` / `null`，默认 `auto` |
| `--video-size H W` | 拼接后的最终尺寸，H/W 必须是 16 的倍数 |
| `--num-frames N` | 默认 `33` |
| `--action-video-freq-ratio N` | 默认 `4` |
| `--global-sample-stride N` | 默认 `1` |
| `--action-key KEY` | 选 `action.<key>` 子特征（可重复） |
| `--state-key KEY` | 选 `observation.state.<key>` 子特征（可重复） |
| `--delta-mask SPEC` | `true/false` 逗号列表，或 `all-false` / `all-true` |
| `--norm-mode MODE` | `min/max`（默认）或 `z-score` |
| `--val-proportion P` | 留出验证集比例，默认 `0` |
| `--context-len N` | 文本编码长度，默认 `128` |
| `--cache-dir PATH` | 文本嵌入缓存目录 |

### 3.3 训练选项

| 参数 | 说明 |
| --- | --- |
| `--gpus N` | 使用卡数（默认自动探测） |
| `--model NAME` | `fastwam` / `fastwam_idm` / `fastwam_joint` / `fastwam_optional_idm` |
| `--epochs N` / `--batch-size N` / `--lr LR` / `--num-workers N` / `--save-every N` | 训练超参 |
| `--resume PATH` | 断点续训（目录或权重 `.pt`） |
| `--norm-stats PATH` | 写入 `data.train.pretrained_norm_stats` |
| `--backbone PATH` | 自定义 ActionDiT backbone |
| `--random-backbone-init` | 跳过 backbone 检查并随机初始化（不推荐） |
| `--online-text-encode` | 不预计算文本嵌入，训练时在线编码 |
| `--zero {1,2}` | DeepSpeed ZeRO stage，默认 `1` |
| `--skip-prereq-check` | 只做配置校验 |
| `--dry-run` | 打印最终训练命令后退出 |
| `--force` | 允许覆盖非自动生成的同名配置 |
| `--extra "k=v k=v"` | 追加 Hydra overrides，位置在启动器默认值之后，可覆盖默认值 |
| `--` | 其后的参数原样透传给训练脚本 |

### 3.4 示例

```bash
# 覆盖训练超参
bash scripts/train_custom.sh --dataset ./data/my_ds --gpus 8 \
  --batch-size 16 --epochs 10 --lr 1e-4

# 指定相机与拼接布局
bash scripts/train_custom.sh --dataset ./data/my_ds --gpus 8 \
  --camera cam_high --camera cam_left_wrist \
  --resize 224 224 --concat horizontal --video-size 224 448

# 不预计算文本嵌入
bash scripts/train_custom.sh --dataset ./data/my_ds --gpus 8 --online-text-encode

# 只生成配置、检查前置条件并打印命令
bash scripts/train_custom.sh --dataset ./data/my_ds --gpus 8 --dry-run

# 追加任意 Hydra 参数
bash scripts/train_custom.sh --dataset ./data/my_ds --gpus 8 \
  --extra "model.compile_training_denoise=true"
```

---

## 4. `prepare_custom_training.py` 设计

该脚本提供四个子命令：

| 子命令 | 作用 |
| --- | --- |
| `generate` | 探测一个或多个数据集，写 data/task 配置；`--print-name` 只把 task 名输出到 stdout，诊断信息走 stderr |
| `import-config` | 导入已有 task/data 配置；`--data-config` 会自动生成薄 task 包装 |
| `override-form` | 把启动器的覆盖项输出为 `key=value` 或 `+key=value` |
| `verify` | 合成 Hydra 配置并做语义与前置条件检查 |

### 4.1 `meta/info.json` 探测规则

**版本**

| `codebase_version` | reader | 额外字段 |
| --- | --- | --- |
| `2.0` / `2.1` | `fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset` | 无 |
| `3.0` | `fastwam.datasets.lerobot3.robot_video_dataset.RobotVideoDataset` | `video_backend: pyav` |

其他版本直接报错。

**相机**

- 只识别 `dtype` 为 `image` 或 `video` 且 key 以 `observation.images.` 开头的特征；
- 去掉前缀作为 `shape_meta.images[*].key`，顺序与 `info.json` 一致；
- 形状优先按 `names`（`height` / `width` / `channels`）判定 HWC，统一转成 CHW：
  `raw_shape = [C, H, W]`；
- 非 3 通道特征告警并跳过；
- `--camera` 可显式选择与排序，`--exclude-camera` 用于排除。

**action / state**

- 优先匹配同名的 `action` / `observation.state`，key 记为 `default`；
- 否则扫描 `action.*` / `observation.state.*` 子特征，key 取后缀；
- 维度取 `shape[0]`，要求一维；多维直接报错；
- 缺少 `observation.state` 硬错误（底层 dataset 没有回退逻辑）；
- 同一层有多个子特征时必须用 `--action-key` / `--state-key` 指定。

**多数据集一致性**

- reader 版本、fps、action/state 的 key 与维度必须一致；
- 相机集合必须一致，否则可用 `--intersect-cameras` 取交集，或用 `--camera` 指定；
- 同名相机分辨率不一致时仅告警。

**episode 长度**

- v2.x 读 `meta/episodes.jsonl` 的 `length`；
- v3.0 读 `meta/episodes/**/*.parquet` 的 `length` 列；
- 采样窗口为 `(num_frames - 1) * global_sample_stride + 1`；
- 最长 episode 小于窗口时硬错误；恰好等于窗口时告警。

### 4.2 默认策略

| 项 | 默认值 |
| --- | --- |
| 采样窗口 | `num_frames=33`、`action_video_freq_ratio=4`（视频帧数 8） |
| resize | 每相机 `224 × 224` |
| 拼接 | 单相机不拼接；多相机 `horizontal` |
| `video_size` | 单相机 `[224, 224]`；水平拼接 `[224, 224*N]` |
| `delta_action_dim_mask` | 全 `false`（delta 语义无法从数据推断） |
| 归一化 | `min/max` |
| 文本缓存 | `./data/text_embeds_cache/autogen_<name>`，`context_len=128` |
| 训练超参 | `batch_size=2`、`num_workers=4`、`num_epochs=1`、`lr=1e-4`、`save_every=100` |
| 模型 | `fastwam` |

`--concat null` 且多相机时必须显式给 `--video-size`；`robotwin` 布局与相机顺序/尺寸强相关，
同样要求显式 `--video-size`。

### 4.3 生成的 data 配置（示例：2 相机 v3.0）

```yaml
# AUTOGENERATED by scripts/prepare_custom_training.py
# datasets: /path/to/FastWAM/data/mybot_v30
# regenerate: bash scripts/train_custom.sh --dataset /path/to/FastWAM/data/mybot_v30 --force
train:
  _target_: fastwam.datasets.lerobot3.robot_video_dataset.RobotVideoDataset
  dataset_dirs:
    - './data/mybot_v30'
  shape_meta:
    images:
      - key: cam_high
        raw_shape: [3, 480, 640]
        shape: [3, 224, 224]
      - key: cam_left_wrist
        raw_shape: [3, 480, 640]
        shape: [3, 224, 224]
    action:
      - key: default
        raw_shape: 7
        shape: 7
    state:
      - key: default
        raw_shape: 8
        shape: 8
  num_frames: 33
  global_sample_stride: 1
  action_video_freq_ratio: 4
  video_size: [224, 448]
  camera_key: null
  val_set_proportion: 0.0
  is_training_set: true
  pretrained_norm_stats: null
  skip_padding_as_possible: false
  concat_multi_camera: "horizontal"
  processor:
    _target_: fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor
    shape_meta: ${data.train.shape_meta}
    num_obs_steps: ${data.train.num_frames}
    num_output_cameras: 2
    action_output_dim: 7
    proprio_output_dim: 8
    delta_action_dim_mask:
      default: [false, false, false, false, false, false, false]
    action_state_transforms: null
    use_stepwise_action_norm: false
    norm_default_mode: 'min/max'
    norm_exception_mode: null
    action_state_merger:
      _target_: fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign
    train_transforms:
      - _target_: fastwam.datasets.lerobot.transforms.image.ToTensor
      - _target_: torchvision.transforms.Resize
        size: [224, 224]
    val_transforms:
      - _target_: fastwam.datasets.lerobot.transforms.image.ToTensor
      - _target_: torchvision.transforms.Resize
        size: [224, 224]
  text_embedding_cache_dir: './data/text_embeds_cache/autogen_mybot_v30'
  use_text_embed_cache: true
  context_len: 128
  video_backend: pyav
```

说明：

- 数据集在仓库内时写 `./` 相对路径，否则写绝对路径，YAML 中一律加引号；
- `pretrained_norm_stats: null` 显式写出，便于后续用 `data.train.pretrained_norm_stats=...`
  直接覆盖，不需要 Hydra 的 `+` 前缀；
- `--val-proportion > 0` 时追加 `val:` 节点，`is_training_set: false` 且**不写**
  `pretrained_norm_stats`（运行时沿用训练节点的统计量）。

### 4.4 生成的 task 配置

```yaml
# @package _global_
# AUTOGENERATED by scripts/prepare_custom_training.py
# datasets: /path/to/FastWAM/data/mybot_v30
# regenerate: bash scripts/train_custom.sh --dataset /path/to/FastWAM/data/mybot_v30 --force
defaults:
  - override /data: autogen_mybot_v30
  - override /model: fastwam
  - _self_

batch_size: 2
num_workers: 4
lr_scheduler_type: "cosine"
learning_rate: 0.0001
num_epochs: 1
max_steps: null
log_every: 10
save_every: 100
eval_every: 50
gradient_accumulation_steps: 1
weight_decay: 1.0e-2
resume: null
```

使用 task 配置而不是直接 `data=... model=...` 的原因：`train_zero1.sh` 只有在参数中出现
`task=` 时才会把输出目录命名为 `runs/<task>/<run_id>`；且 `precompute_text_embeds.py`
也需要 task 配置同时提供 data 与 model。

### 4.5 语义校验

`generate` 与 `verify` 共用同一套约束：

1. `num_frames - 1` 能被 `action_video_freq_ratio` 整除；
2. `(num_frames - 1) / action_video_freq_ratio` 是 4 的倍数（VAE tokenization 约束）；
3. `video_size` 的 H、W 均为 16 的正整数倍；
4. 每个相机的 `shape[-2:]` 与 `Resize.size` 一致；
5. `processor.num_output_cameras == len(shape_meta.images)`；
6. `processor.action_output_dim` / `proprio_output_dim` 与 shape_meta 求和一致；
7. `delta_action_dim_mask` 的 key 集合与 action keys 一致，每组长度与动作维度一致；
8. 所有 data 节点的 `context_len` 一致。

### 4.6 前置条件检查

**ActionDiT backbone**

- compose 后读取 `model.action_dit_pretrained_path`，相对路径按仓库根解析；
- `skip_dit_load_from_pretrain: true` 或路径为空时跳过；
- 缺失时打印：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/<model>.yaml \
  --output <backbone 路径> \
  --device cuda --dtype bfloat16
```

并以退出码 `2` 结束。`--random-backbone-init` 改为追加
`model.action_dit_pretrained_path=null model.skip_dit_load_from_pretrain=true`。

**文本嵌入缓存**

- 仅对 `use_text_embed_cache: true` 的 dataset 节点检查；
- 从 `meta/tasks.jsonl`（v3.0 回退 `meta/tasks.parquet`）读任务文本，按
  `sha256(DEFAULT_PROMPT.format(task=t))` 计算期望文件名：
  `<hash>.t5_len<context_len>.<enc_id>.pt`，其中 `enc_id` 由 `model_id`
  最后一段转小写并仅保留字母数字（默认 `Wan-AI/Wan2.2-TI2V-5B` → `wan22ti2v5b`）；
- 缺失时打印单卡与 `torchrun` 两条命令以及免缓存方案，并退出 `2`：

```bash
python scripts/precompute_text_embeds.py task=<task>
torchrun --standalone --nproc_per_node=<gpus> scripts/precompute_text_embeds.py task=<task>
# 或：bash scripts/train_custom.sh --task <task> --online-text-encode
```

`--online-text-encode` 会跳过缓存检查，并追加
`data.train.use_text_embed_cache=false model.load_text_encoder=true`。

### 4.7 覆盖与幂等

- 生成文件首行带 `AUTOGENERATED by scripts/prepare_custom_training.py` 标记；
- 重新生成时：内容相同 → `unchanged`；内容不同且现有文件带标记 → 覆盖；
  现有文件无标记 → 拒绝，除非 `--force`；
- 写入采用临时文件 + `os.replace` 的原子方式；
- slug 规则：`autogen_<sanitize(name)>`，只保留 `[a-z0-9_]`，最长 64 字符；
- 同一 slug 但 `# datasets:` 记录的路径不同时，追加路径列表 sha1 前 6 位，
  例如 `autogen_fwtest_v30_7fea2d`；
- 导入外部配置时，目标已存在且内容不同则要求 `--force`；内容相同直接复用。

---

## 5. `train_custom.sh` 流程

1. `set -euo pipefail`，记录 `ORIG_PWD`，解析仓库根并 `cd "$REPO_ROOT"`
   （配置中的相对路径都按仓库根解释）；
2. 数组式参数解析，路径按调用时的工作目录转成绝对路径，避免空格拆分；
3. 校验 `--dataset` / `--task` / `--task-file` / `--data-config` 恰好提供一个；
4. 解析 GPU 数并校验可见性；
5. 调 `generate` 或 `import-config` 得到 `TASK_NAME`；
6. 把 `--epochs` / `--batch-size` / `--lr` / `--num-workers` / `--save-every` /
   `--resume` / `--norm-stats` / `--online-text-encode` 等转成覆盖项，经
   `override-form` 规范化为 `key=value` 或 `+key=value`；
7. 调 `verify`，按退出码中止；`--dry-run` 用 `printf '%q'` 打印转义后的最终命令；
8. `exec bash scripts/train_zero${ZERO}.sh "$GPUS" "task=$TASK_NAME" "${OVERRIDES[@]}"`。

`--extra` 与 `--` 透传项排在启动器默认值之后，因此可以覆盖启动器生成的默认值。

---

## 6. 其他改动

### 6.1 `scripts/precompute_text_embeds.py`

新增 `_read_task_texts()`：`meta/tasks.jsonl` 存在时按原逻辑读取；否则回退到
LeRobot 3.0 的 `meta/tasks.parquet`（优先 `task` 列，没有该列时使用 DataFrame
index，与 v3 reader 的 `tasks.iloc[idx].name` 一致）；两者都不存在时报错并列出两个路径。

### 6.2 `docs/custom_data_tutorial_zh.md`

顶部新增「一键启动（推荐）」小节，包含四种入口、常用选项、自动生成文件位置、
前置检查行为以及 `--dry-run` / `--online-text-encode` / `--extra` 的用法。
原 1–9 节的手动流程保留，供需要精细控制的场景使用。

---

## 7. 验收结果

验证环境：Python 3.12、Hydra 1.3.2、OmegaConf 2.3.0、pandas 3.0.6、
pyarrow 25.0.1、torch 2.7.1+cu126。该 shell 未安装 `fastwam`，涉及 Hydra
合成的检查通过 `PYTHONPATH=src` 完成（容器内已 `pip install -e .`，无需该变量）。

### 7.1 静态检查

- `python -m py_compile scripts/prepare_custom_training.py` 通过；
- `bash -n scripts/train_custom.sh` 通过；
- `prepare_custom_training.py` 的四个子命令 `--help` 均可执行。

### 7.2 正例（`/tmp` 下人工构造的 LeRobot fixture）

| 场景 | 结果 |
| --- | --- |
| v3.0 fixture 生成 | 使用 `fastwam.datasets.lerobot3...`；相机 `[3, 480, 640]`；`video_size [224, 448]`；action/state 7/8；写入 `video_backend: pyav` |
| v2.1 fixture 生成 | 使用 `fastwam.datasets.lerobot...`；不写 `video_backend` |
| 重复生成 | 输出 `unchanged`，幂等 |
| 多子特征 action/state（`arm` + `gripper`） | 生成两条 shape_meta 条目，按 key 写 mask，维度求和 |
| `--val-proportion > 0` | 追加 `val:` 节点，`is_training_set: false`，不写 `pretrained_norm_stats` |
| 同名 slug 不同数据集 | 自动加路径 hash 后缀，如 `autogen_fwtest_v30_7fea2d` |
| `--dry-run` | 输出最终命令，例如 `bash scripts/train_zero1.sh 1 task=autogen_finaltest ...` |
| 假 backbone + 假文本缓存 | `config + prerequisites OK` |
| `--online-text-encode` 导入 `robotwin` 配置 | 正确输出 `+data.train.use_text_embed_cache=false`（该配置原本没有此 key） |
| `--dataset` 传参后 `override-form` | 已有 key 输出 `key=value`，缺失 key 输出 `+key=value` |

### 7.3 前置条件缺失

| 场景 | 结果 |
| --- | --- |
| 缺少 ActionDiT backbone | 退出码 `2`，打印 `preprocess_action_dit_backbone.py` 完整命令 |
| 缺少文本缓存 | 退出码 `2`，打印单卡 / `torchrun` / `--online-text-encode` 三种处理方式 |

### 7.4 负例（均退出码 `1`，信息可操作）

- `delta_action_dim_mask` 长度与动作维度不一致；
- `video_size` 不是 16 的倍数；
- `num_frames` / `action_video_freq_ratio` 不满足整除与 4 的倍数约束；
- 缺少 `observation.state`；
- 多个 action 子特征但未指定 `--action-key`；
- 多相机且 `--concat null` 但未提供 `--video-size`；
- 数据集目录不存在或缺少 `meta/info.json`；
- 多数据集 reader 版本 / fps / 相机集合 / 维度不一致；
- 最长 episode 小于采样窗口。

测试生成的 `configs/**/autogen_*` 文件已在验证后清理，仓库中不残留测试配置。

### 7.5 未验证项与已知限制

- **未执行真实训练**：验证环境没有符合要求的数据集、GPU 与 ActionDiT backbone，
  因此只验证到「配置生成 + 合成 + 前置条件检查」这一层。真实训练需要在容器内
  按下面第 8 节执行。
- `delta_action_dim_mask` 无法从数据推断，默认全 `false`；动作是否为 delta
  需要用户用 `--delta-mask` 明确指定。
- `concat_multi_camera: robotwin` 与相机顺序/尺寸强相关，不做自动推导。
- 多数据集只做一致性校验，不做自动重采样；fps 不同会直接报错。
- 导入 shell 时偶发 `datasets` / `multiprocess` 在解释器退出阶段的
  `resource_tracker` 告警（Python 3.12 环境问题），不影响退出码与结果。
- 生成的 `autogen_*` 配置未加入 `.gitignore`，是否提交由用户决定。

---

## 8. 真实训练冒烟测试

在有数据与 GPU 的环境中按以下顺序执行：

```bash
# 1) 生成 backbone（只需一次）
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16

# 2) 先用 --dry-run 确认配置与命令
bash scripts/train_custom.sh --dataset /path/to/my_dataset --gpus 1 --dry-run

# 3) 按提示预计算文本嵌入（或加 --online-text-encode 跳过）
python scripts/precompute_text_embeds.py task=autogen_my_dataset

# 4) 单卡小规模冒烟
bash scripts/train_custom.sh --dataset /path/to/my_dataset --gpus 1 \
  --batch-size 1 --num-workers 0
```

冒烟通过的标准：

1. `runs/autogen_<slug>/<timestamp>/config.yaml` 中 action/state 维度、相机列表与
   `video_size` 与数据集一致；
2. 训练能正常前向/反向并产出第一个 checkpoint；
3. 恢复训练（`--resume`）能正确加载。

---

## 9. 训练启动脚本的关系：`train_zero1.sh` / `train_zero2.sh` 与 `train_custom.sh`

### 9.1 `train_zero1.sh` / `train_zero2.sh` 是什么

它们是**分布式启动器**，本身不包含模型或训练逻辑；真正的训练逻辑在
`scripts/train.py`。两个脚本除 accelerate 配置文件和一处 usage 文案外完全一致，
区别只在 DeepSpeed ZeRO stage：

| | `train_zero1.sh` | `train_zero2.sh` |
| --- | --- | --- |
| accelerate 配置 | `accelerate_zero1_ds.yaml` | `accelerate_zero2_ds.yaml` |
| DeepSpeed JSON | `scripts/ds_configs/ds_zero1_config.json` | `scripts/ds_configs/ds_zero2_config.json` |
| ZeRO stage | 1：只切分优化器状态 | 2：切分优化器状态 + 梯度 |
| 显存占用 | 较高 | 较低 |
| 通信开销 | 较低 | 较高 |
| CPU offload | 无 | 无 |

调用方式：

```bash
bash scripts/train_zero1.sh <nproc_per_node> [hydra overrides...]
```

它负责：

1. 第一个位置参数是**每台机器的进程数（GPU 数）**，其余参数原样传给
   `scripts/train.py`；
2. 从 `task=<name>` 或 `--config-name` 解析出 `TASK_BASENAME`；
3. 生成或同步 `RUN_ID`：
   - 单机：`date +%Y-%m-%d_%H-%M-%S`；
   - 多机：通过 `torch.distributed.TCPStore` 让所有节点拿到同一个 `RUN_ID`；
   - 也可用环境变量 `RUN_ID=...` 固定；
4. 注入两个 Hydra 参数：

   ```bash
   output_dir=./runs/${TASK_BASENAME}/${RUN_ID}
   wandb.name=${TASK_BASENAME}
   ```

5. 调用：

   ```bash
   accelerate launch --config_file scripts/accelerate_configs/accelerate_zero{1,2}_ds.yaml \
     --num_processes <nproc> scripts/train.py ...
   ```

多机相关环境变量：`NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`。

> 这两个脚本**不生成任何配置**。如果 `configs/task/<name>.yaml` 不存在，
> 或配置内容与数据不匹配，它们仍会直接启动，然后在 Hydra 或 dataloader 阶段报错。

### 9.2 与 `train_custom.sh` 的差异

二者是**前后端关系**，不是替代关系：

```text
train_custom.sh（前端：生成配置 + 校验）
        │  exec
        ▼
train_zero{1,2}.sh（后端：accelerate + DeepSpeed + RUN_ID + 输出目录）
        │
        ▼
scripts/train.py（真正的训练逻辑）
```

| 对比项 | `train_custom.sh` | `train_zero1.sh` / `train_zero2.sh` |
| --- | --- | --- |
| 作用 | 配置生成、导入、校验、前置检查 | 进程启动、DS 配置、RUN_ID、输出目录 |
| 输入 | 数据集路径 / task 配置 / data 配置 | 已存在的 task 配置 + Hydra overrides |
| 是否探测 `meta/info.json` | 是 | 否 |
| 是否生成 `configs/**/autogen_*.yaml` | 是 | 否 |
| 是否检查 backbone / 文本缓存 | 是（缺失退出码 `2`） | 否 |
| ZeRO 选择 | `--zero {1,2}` | 由脚本本身决定 |
| 多卡 / 多机 / RUN_ID | 透传环境变量，最终交给后端 | 自己实现 |
| 输出目录 | 最终仍由后端生成 | `runs/<task>/<run_id>` |

换句话说：**`train_custom.sh` 正是通过调用这两个脚本完成训练的**。它最后执行的是：

```bash
exec bash scripts/train_zero${ZERO}.sh "$GPUS" "task=$TASK_NAME" "${OVERRIDES[@]}"
```

`--zero 1`（默认）走 `train_zero1.sh`，`--zero 2` 走 `train_zero2.sh`。

### 9.3 如何配合使用

**推荐：只调用 `train_custom.sh`**

```bash
# 默认 ZeRO-1
bash scripts/train_custom.sh --dataset ./data/my_ds --gpus 8

# 显存不够时切 ZeRO-2
bash scripts/train_custom.sh --dataset ./data/my_ds --gpus 8 --zero 2

# 先看它会调用哪条后端命令
bash scripts/train_custom.sh --dataset ./data/my_ds --gpus 8 --zero 2 --dry-run
# [dry-run] bash scripts/train_zero2.sh 8 task=autogen_my_ds ...
```

**已有配置时：直接调用后端**

```bash
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4 batch_size=16
bash scripts/train_zero2.sh 8 task=libero_uncond_2cam224_1e-4 batch_size=16
```

任务名、输出目录、wandb 命名都由 `task=` 决定。

**透传额外 Hydra 参数**

`train_custom.sh` 提供两种方式，最终都会落到后端命令的末尾：

```bash
# --extra：等价的 key=value 列表
bash scripts/train_custom.sh --dataset ./data/my_ds --gpus 8 \
  --extra "model.compile_training_denoise=true"

# -- 之后：原样透传（最后追加，优先级最高）
bash scripts/train_custom.sh --dataset ./data/my_ds --gpus 8 \
  -- data.train.num_workers=8
```

**多机与固定 RUN_ID**

这些环境变量由后端脚本读取，`train_custom.sh` 通过 `exec` 原样传递：

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
  bash scripts/train_custom.sh --dataset ./data/my_ds --gpus 8

# 固定实验名
RUN_ID=my_exp_001 bash scripts/train_custom.sh --dataset ./data/my_ds --gpus 8
```

多机场景下每个节点都需要能访问同一份数据和配置；最稳妥的做法是在每个节点各跑一次
`generate`（输出是确定性的，文件名相同），或提前把生成的
`configs/{data,task}/autogen_*.yaml` 同步到各节点。

### 9.4 ZeRO stage 选择经验

- 先用 **ZeRO-1**（默认）：通信少、吞吐高，多数单机 8 卡场景够用；
- 出现 OOM 且不想减小 batch 时切 **ZeRO-2**：再省一份梯度显存，代价是
  all-gather 通信增加；
- 两个配置都没有 CPU offload。如果 ZeRO-2 仍然 OOM，应调小 `batch_size`
  或开启 gradient checkpointing；当前脚本未暴露 ZeRO-3。
