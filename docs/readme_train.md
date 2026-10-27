
### 拷贝文件

``` bash
./obsutil  cp   obs://egowm/data_h00672373/fastwam   /mnt/sfs_turbo/h00672373/docker_dir/checkpoints    -r  -f
```

### 创建容器
```bash
IMAGE=fastwam:cu126    HOST_DATA_DIR=/mnt/sfs_turbo/h00672373/train_data   bash docker/run_container.sh


IMAGE=fastwam:cu126 \
REPO_ROOT=/mnt/sfs_turbo/h00672373/docker_dir \
HOST_DATA_DIR=/mnt/sfs_turbo/h00672373/train_data  \
EXTRA_DOCKER_ARGS="-v /mnt/sfs_turbo/h00672373/github/FastWAM:/workspace/FastWAM" \
GPUS='"device=4,5,6,7"'  \
./docker/run_container.sh


IMAGE=fastwam:cu126 \
HOST_DATA_DIR=/mnt/sfs_turbo/h00672373/train_data \
HOST_CKPT_DIR=/mnt/sfs_turbo/h00672373/docker_dir/checkpoints \
HOST_RUNS_DIR=/mnt/sfs_turbo/h00672373/docker_dir/runs \
HOST_EVAL_DIR=/mnt/sfs_turbo/h00672373/docker_dir/evaluate_results \
HOST_HF_CACHE=/mnt/sfs_turbo/h00672373/docker_dir/hf_cache \
EXTRA_DOCKER_ARGS="-v /mnt/sfs_turbo/h00672373/github/FastWAM:/workspace/FastWAM" \
GPUS='"device=0,1,2,3"' \
bash docker/run_container.sh



IMAGE=fastwam:cu126 \
HOST_DATA_DIR=/mnt/sfs_turbo/h00672373/train_data \
HOST_CKPT_DIR=/mnt/sfs_turbo/h00672373/docker_dir/checkpoints \
HOST_RUNS_DIR=/mnt/sfs_turbo/h00672373/docker_dir/runs \
HOST_EVAL_DIR=/mnt/sfs_turbo/h00672373/docker_dir/evaluate_results \
HOST_HF_CACHE=/mnt/sfs_turbo/h00672373/docker_dir/hf_cache \
EXTRA_DOCKER_ARGS="-v /mnt/sfs_turbo/h00672373/github/FastWAM:/workspace/FastWAM" \
GPUS='"device=4,5,6,7"' \
MAME="4-7" \
bash docker/run_container.sh

```

### 下载模型权重
```bash
export MODELSCOPE_ENDPOINT=https://www.modelscope.ai  &&  modelscope download --model Wan-AI/Wan2.1-T2V-1.3B   --local_dir ./Wan2.1-T2V-1.3B


export HF_ENDPOINT=https://hf-mirror.com  &&   ./hfd.sh  google/umt5-xxl   --local-dir  /mnt/sfs_turbo/h00672373/docker_dir/checkpoints/umt5-xxl   --tool aria2c -x 4
```

### 数据转lerobotv30

``` bash
python -m   \
    --repo-id=lerobot/hs \
    --root=/mnt/sfs_turbo/h00672373/train_data/hs_data_v30/lerobot_white_0324 \
    --push-to-hub=false

python -m lerobot.scripts.convert_dataset_v21_to_v30 \
    --repo-id=lerobot/hs \
    --root=/mnt/sfs_turbo/h00672373/train_data/guodi_sanitized/wrc3 \
    --push-to-hub=false


python src/lerobot/scripts/convert_dataset_v21_to_v30.py \
    --repo-id=lerobot/pusht \
    --root=/path/to/local/dataset/directory
```

### 数据集列名修复（meta 与 parquet 不一致）

```bash
cd /workspace/FastWAM

# 训练报 ArrowInvalid: No match for FieldRef.Name(frame_index)
#   -> meta/info.json 是 frame_index，data/*.parquet 里是 frame_idx
# 先只看计划（默认不写文件）
python scripts/fix_dataset_columns.py \
    --dataset /workspace/data/hs_data_v30/lerobot_charger_0324 \
    --dataset /workspace/data/hs_data_v30/lerobot_white_0324

# 确认后原地改列名（只改 data/*.parquet，不动 videos/ 与 meta/）
python scripts/fix_dataset_columns.py \
    --dataset /workspace/data/hs_data_v30/lerobot_charger_0324 \
    --dataset /workspace/data/hs_data_v30/lerobot_white_0324 \
    --write
```

### 数据集字段修改

```
scripts/adapt_dataset_for_training.py（就是这个，--rename 改列名，--fps 改帧率）。

# 只改列名：actions -> action
python scripts/adapt_dataset_for_training.py \
    --src /workspace/data/ds_b \
    --dst /workspace/data/ds_b_renamed \
    --copy-videos \
    --rename actions=action

# 改列名 + 降 fps（可同时 --rename 多次）
python scripts/adapt_dataset_for_training.py \
    --src /workspace/data/ds_b \
    --dst /workspace/data/ds_b_15fps_action \
    --fps 15 \
    --rename actions=action \
    --rename observation.state.raw=observation.state

# 先看计划，不落盘
... --dry-run
... --dst 之后是不是要离开这台机器的这个路径。要归档、要发给别人、要换容器挂载点，就用 --copy-videos；只是本机快速试一版、数据集又很大，就用软链。
```


### 训练脚本

> **顺序：第 0 步 → 第 1 步 → 第 2 步 → 第 3 步（dry-run）→ 第 4 步（正式训练）。**
>
> 第 1、3、4 步的 `--dataset / --action-feature / --camera` 必须完全一致：
> `train_custom.sh` 每次都会按这些参数重新生成
> `configs/data|task/autogen_<name>.yaml`，参数不一致会得到不同的配置。
>
> 本数据集的动作用列名是 **`actions`**（不是 LeRobot 标准的 `action`），
> 因此必须带 `--action-feature actions`，否则生成器会报
> `数据集中找不到 action 特征 'action'`。
> （该参数由当前分支的代码提供；容器通过 `EXTRA_DOCKER_ARGS` 挂载宿主机仓库，
> 改动即时生效，无需重建镜像。）

```bash
cd /workspace/FastWAM

# 0) 一次性：生成 ActionDiT backbone
#    会同时加载 DiT(10G) + 文本编码器(11G) + VAE，单张 H800 约 25 GB。
#    这步不会触发下载（前提是权重已按上面的方式放好）。
CUDA_VISIBLE_DEVICES=0 python scripts/preprocess_action_dit_backbone.py \
    --model-config configs/model/fastwam.yaml \
    --output /workspace/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
    --device cuda --dtype bfloat16
```

```bash
# 1) 一次性：按数据集生成 data/task 配置，并打印任务名
#    --camera 的顺序决定水平拼接后的画面顺序（front 在左，wrist 在右）
cd /workspace/FastWAM
python scripts/prepare_custom_training.py generate \
    --dataset /workspace/data/lerobot_v30_flexiv_hz_union_v1 \
    --action-feature actions \
    --camera front --camera wrist \
    --print-name
# 输出（也是下面两步要用的 task 名）：
#   autogen_lerobot_v30_flexiv_hz_union_v1
```

```bash
# 2) 一次性：预计算文本嵌入缓存（依赖第 1 步生成的 task 配置）
cd /workspace/FastWAM
python scripts/precompute_text_embeds.py \
    task=autogen_lerobot_v30_flexiv_hz_union_v1

# 本数据集有 4 条任务文本 -> 生成 4 个 .pt 文件，产物在
#   /workspace/FastWAM/data/text_embeds_cache/autogen_lerobot_v30_flexiv_hz_union_v1/
# 宿主机对应 .../github/FastWAM/data/...（.gitignore 已忽略 data/*）。
#
# 多卡加速（可选，4 条文本单卡也很快）：
#   torchrun --standalone --nproc_per_node=4 scripts/precompute_text_embeds.py \
#       task=autogen_lerobot_v30_flexiv_hz_union_v1
```

```bash
# 3) 先 dry-run：检查 backbone + 文本缓存，通过后打印真正要执行的命令
cd /workspace/FastWAM
BACKBONE=/workspace/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt

bash scripts/train_custom.sh \
    --dataset /workspace/data/lerobot_v30_flexiv_hz_union_v1 \
    --action-feature actions \
    --camera front --camera wrist \
    --gpus 4 \
    --backbone "$BACKBONE" \
    --dry-run

# 通过时最后一行形如：
# [dry-run] bash scripts/train_zero1.sh 4 task=autogen_lerobot_v30_flexiv_hz_union_v1 ...
```

```bash
# 4) 正式训练（与第 3 步参数完全一致，只去掉 --dry-run）
cd /workspace/FastWAM
BACKBONE=/workspace/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt

bash scripts/train_custom.sh \
    --dataset /workspace/data/lerobot_v30_flexiv_hz_union_v1 \
    --action-feature actions \
    --camera front --camera wrist \
    --gpus 4 \
    --backbone "$BACKBONE"
```

#### 训练日志

正式训练（第 4 步 / 多数据集第 6 步）的输出同时写终端和日志文件：

```text
/workspace/FastWAM/log/<yyyy-mm-dd-hhmm>_<model>_<task>.log
# 例：/workspace/FastWAM/log/2026-10-23-1105_fastwam_autogen_lerobot_v30_flexiv_hz_union_v1.log
```

- `<model>` 依次取 `--model`、`--extra "model=..."`、task 配置里的 `override /model:`，都没有时为 `unknown`
- 时间戳精确到分钟，文件名无空格和冒号；同一分钟内重跑同名任务会**追加**写，每次以 `=====` 分隔并记录启动时间与完整命令；换分钟即新文件
- 换目录：`--log-dir PATH`；不要日志：`--no-log`；`--dry-run` 不写日志
- 宿主机对应 `.../github/FastWAM/log/`（`*.log` 已被 .gitignore 忽略）
- 实时跟踪：`tail -f /workspace/FastWAM/log/<文件名>`

#### 显存不够：梯度累计

`--batch-size` 是**每卡**的 micro-batch；显存不够时降它、用 `--grad-accum` 把有效 batch
补回来，不必改卡数：

```bash
# 有效 batch = batch-size × 卡数 × grad-accum
bash scripts/train_custom.sh ... --gpus 4 --batch-size 2 --grad-accum 4   # 有效 32
```

#### checkpoint / 磁盘配额（报 Errno 122）

每个 checkpoint ≈ 106 GB（weights 12 GB + ZeRO-1 state 94 GB）。
`configs/train.yaml` 默认 `save_every: 100` + `keep_last_n: 2`，盘占用有上限，
但存得太密会反复卡住训练，长任务仍建议拉长间隔：

```bash
# 拉长保存间隔 + 只保留最近 2 个（configs/train.yaml 默认 keep_last_n: 2）
bash scripts/train_custom.sh ... --save-every 2000 --keep-last 2

# 固定总步数（默认 null = 按 epochs × 样本数/(batch×卡数) 自动算，0 = 恢复自动）
bash scripts/train_custom.sh ... --max-steps 20000

# 绝对动作转 delta action：默认转 0-5 维（位姿），夹爪/力/力矩保持绝对
bash scripts/train_custom.sh ... --delta-action
# 只转位置 3 维（旋转是 Euler 角时建议这样，差值在 ±π 会回绕）
bash scripts/train_custom.sh ... --delta-action-dims 0,1,2
```

> `--delta-action` 会在 processor 里自动挂 `RelativeActionTransform`，归一化统计在
> delta 空间重算；启用后不要复用旧的 `--norm-stats`。

报 `Disk quota exceeded` 时（`df` 仍显示有空间，SFS Turbo 目录配额不体现在 df 里）：

```bash
cd /workspace/FastWAM/runs/<task>/<run_id>/checkpoints
ls -d state/step_* | sort | head -n -1 | xargs -r rm -rf   # 留最新 state
ls weights/step_*.pt | sort | head -n -1 | xargs -r rm -f  # 留最新权重
# 保存中断的 state/step_XXXXXX 是半截文件，不能用于 --resume，直接删
```

`train_custom.sh` 启动前会往 `<repo>/runs` 试写 64 MiB 探针（`WRITE_PROBE_MB`），
写不进去直接退出。

#### 跳过第 2 步时

dry-run 会报：

```text
error: 训练前置条件缺失：
  文本嵌入缓存不完整：缺少 4 个文件（示例 .../text_embeds_cache/autogen_lerobot_v30_flexiv_hz_union_v1/<hash>.t5_len128.wan22ti2v5b.pt）
```

两种解决方式（二选一）：

- 按第 2 步生成缓存（推荐，训练启动更快、不占额外显存）；
- 改在线编码（不建缓存，训练时动态跑文本编码器）：

```bash
bash scripts/train_custom.sh \
    --dataset /workspace/data/lerobot_v30_flexiv_hz_union_v1 \
    --action-feature actions \
    --camera front --camera wrist \
    --gpus 4 \
    --backbone "$BACKBONE" \
    --online-text-encode
```



### 训练脚本-多数据集

> 原理与细节见 `docs/custom_data_tutorial_zh.md` 第 10 节；下面只列执行命令，按 0 → 6 顺序。

解决多个数据集可能存在的 数据集本身 meta 和 parquet 的列名不一致【前面有说明】
```
# 1) 先看计划（默认不写文件）
python scripts/fix_dataset_columns.py \
    --dataset /workspace/data/hs_data_v30/lerobot_charger_0324 \
    --dataset /workspace/data/hs_data_v30/lerobot_white_0324

# 预期输出：
#   == /workspace/data/hs_data_v30/lerobot_charger_0324
#      数据文件: 1 个
#      改名: frame_idx -> frame_index（1 个文件）

# 2) 确认后原地改（只重写 data/*.parquet，两个文件共 16 MB；videos/ 和 meta/ 不动）
python scripts/fix_dataset_columns.py \
    --dataset /workspace/data/hs_data_v30/lerobot_charger_0324 \
    --dataset /workspace/data/hs_data_v30/lerobot_white_0324 \
    --write
```

```bash
cd /workspace/FastWAM

# 0) 校验多个数据集能否混合训练（0 可混合 / 1 有冲突 / 2 参数错误）
python scripts/check_data_merge_train.py \
    --dataset /workspace/data/ds_a \
    --dataset /workspace/data/ds_b \
    --action-feature actions


### 数据集 hs
python scripts/check_data_merge_train.py \
    --dataset /workspace/data/hs_data_v30/lerobot_charger_0324 \
    --dataset /workspace/data/hs_data_v30/lerobot_white_0324 \
    --action-feature actions
    # --dataset /workspace/data/lerobot_v30_flexiv_hz_union_v1 \
    # --dataset /workspace/data/hs_data_v30/hs_charger_white_da3_7d_20261022 \


### 数据集-guodi
python scripts/check_data_merge_train.py \
    --dataset /workspace/data/guodi_sanitized_v30/wrc1-1 \
    --dataset /workspace/data/guodi_sanitized_v30/wrc2 \
    --dataset /workspace/data/guodi_sanitized_v30/wrc3 \
    --action-feature action

### 数据集-ur5
python scripts/check_data_merge_train.py \
    --dataset /workspace/data/ur5e_v30/pick_place_block_150_train_v3.0 \
    --dataset /workspace/data/ur5e_v30/pick_place_three_blocks_150_train_v3.0 \
    --dataset /workspace/data/ur5e_v30/pick_place_pens_blocks_170-merge_train_v3.0 \
    --camera  wrist --camera third_person \
    --state-feature observation.state \
    --action-feature action


python scripts/check_data_merge_train.py \
    --dataset /workspace/data/lerobot_v30_flexiv_hz_union_v1  \
    --camera  front --camera wrist  \
    --state-feature observation.state \
    --action-feature actions

#####################################################################

# 1) 有冲突时适配数据：统一 fps / 列名，输出到新目录，不改原数据
python scripts/adapt_dataset_for_training.py \
    --src /workspace/data/ds_b \
    --dst /workspace/data/ds_b_15fps \
    --fps 15 \
    --dry-run          # 确认计划后去掉 --dry-run
# 列名不一致时追加 --rename actions=action，
# 并把后续所有命令的 --action-feature 改成 action（所有数据集要统一）

# 2) 重新校验
python scripts/check_data_merge_train.py \
    --dataset /workspace/data/ds_a \
    --dataset /workspace/data/ds_b_15fps \
    --action-feature actions


# 3) 生成多数据集配置（task 名 autogen_<name>）
python scripts/prepare_custom_training.py generate \
    --dataset /workspace/data/ds_a \
    --dataset /workspace/data/ds_b_15fps \
    --name mixed_ab \
    --action-feature actions \
    --camera front --camera wrist \
    --print-name        # -> autogen_mixed_ab

### hs
python scripts/prepare_custom_training.py generate \
    --dataset /workspace/data/hs_data_v30/lerobot_charger_0324 \
    --dataset /workspace/data/hs_data_v30/lerobot_white_0324 \
    --name mixed_charger_white \
    --action-feature actions \
    --print-name 
    # --dataset /workspace/data/lerobot_v30_flexiv_hz_union_v1 \

### 国地
python scripts/prepare_custom_training.py \
    --dataset /workspace/data/guodi_sanitized_v30/wrc1-1 \
    --dataset /workspace/data/guodi_sanitized_v30/wrc2 \
    --dataset /workspace/data/guodi_sanitized_v30/wrc3 \
    --name mixed_guodi-wrc \
    --action-feature action \
    --print-name 

### ur5e
python scripts/prepare_custom_training.py \
    --dataset /workspace/data/ur5e_v30/pick_place_block_150_train_v3.0 \
    --dataset /workspace/data/ur5e_v30/pick_place_three_blocks_150_train_v3.0 \
    --dataset /workspace/data/ur5e_v30/pick_place_pens_blocks_170-merge_train_v3.0 \
    --camera  wrist --camera third_person \
    --state-feature observation.state \
    --action-feature action \
    --name mixed_ur5 \
    --print-name 

### xingsong
python scripts/prepare_custom_training.py \
    --dataset /workspace/data/xinsong-resort-wam_join-gripper  \
    --camera  head --camera left_wrist  --camera right_wrist \
    --state-feature observation.state \
    --action-feature action \
    --name mixed_xingsong-resort-join-gripper \
    --print-name 


### fleixiv
python scripts/prepare_custom_training.py \
    --dataset /workspace/data/lerobot_v30_flexiv_hz_union_v1  \
    --camera  front --camera wrist  \
    --state-feature observation.state \
    --action-feature actions \
    --name mixed_lerobot_v30_flexiv_hz_union_v1 \
    --print-name 



# 4) 预计算文本嵌入（覆盖所有数据集的任务文本，合并去重）
python scripts/precompute_text_embeds.py task=autogen_mixed_ab

python scripts/precompute_text_embeds.py task=autogen_mixed_flexiv_charger_white

python scripts/precompute_text_embeds.py task=autogen_mixed_charger_white

python scripts/precompute_text_embeds.py task=autogen_mixed_guodi_wrc

python scripts/precompute_text_embeds.py task=autogen_mixed_ur5

python scripts/precompute_text_embeds.py  task=autogen_mixed_xingsong_resort_join_gripper 

python scripts/precompute_text_embeds.py    task=autogen_mixed_lerobot_v30_flexiv_hz_union_v1 

# 5) dry-run
BACKBONE=/workspace/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
bash scripts/train_custom.sh \
    --dataset /workspace/data/ds_a \
    --dataset /workspace/data/ds_b_15fps \
    --name mixed_ab \
    --action-feature actions \
    --camera front --camera wrist \
    --gpus 4 --backbone "$BACKBONE" --dry-run


BACKBONE=/workspace/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
bash scripts/train_custom.sh \
    --task autogen_mixed_flexiv_charger_white \
    --action-feature actions \
    --camera front --camera wrist \
    --gpus 4 --backbone "$BACKBONE" --dry-run


BACKBONE=/workspace/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
bash scripts/train_custom.sh \
    --task autogen_mixed_charger_white \
    --action-feature actions \
    --camera front --camera wrist \
    --epochs 2  \
    --batch-size 8 \
    --gpus 4 --backbone "$BACKBONE" --dry-run



# 6) 正式训练（与第 5 步参数一致，去掉 --dry-run）
bash scripts/train_custom.sh \
    --dataset /workspace/data/ds_a \
    --dataset /workspace/data/ds_b_15fps \
    --name mixed_ab \
    --action-feature actions \
    --camera front --camera wrist \
    --gpus 4 --backbone "$BACKBONE"


BACKBONE=/workspace/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
bash scripts/train_custom.sh \
    --task autogen_mixed_charger_white \
    --action-feature actions \
    --camera front --camera wrist \
    --max-steps  20000 \
    --batch-size 8 \
    --grad-accum 4 \
    --gpus 4 --backbone "$BACKBONE"



BACKBONE=/workspace/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
bash scripts/train_custom.sh \
    --task autogen_mixed_guodi_wrc \
    --action-feature action \
    --max-steps  20000 \
    --batch-size 4 \
    --grad-accum 2 \
    --gpus 4  \
    --norm-stats /mnt/sfs_turbo/h00672373/docker_dir/runs/autogen_mixed_guodi_wrc/2026-10-24_07-27-47/dataset_stats.json \
    --backbone "$BACKBONE" --dry-run

### ur5e
BACKBONE=/workspace/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
bash scripts/train_custom.sh \
    --task autogen_mixed_ur5 \
    --action-feature action \
    --max-steps  40000 \
    --batch-size 4 \
    --grad-accum 2 \
    --gpus 4  \
    --resume  /workspace/FastWAM/runs/autogen_mixed_ur5/2026-10-26_02-21-10/checkpoints/state/step_020000  \
    --backbone "$BACKBONE" --dry-run



### xingsong
BACKBONE=/workspace/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
bash scripts/train_custom.sh \
    --task autogen_mixed_xingsong_resort_join_gripper \
    --action-feature action \
    --max-steps  30000 \
    --batch-size 4 \
    --grad-accum 2 \
    --gpus 4  \
    --backbone "$BACKBONE" --dry-run


## 非夕
BACKBONE=/workspace/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
bash scripts/train_custom.sh \
    --task autogen_mixed_lerobot_v30_flexiv_hz_union_v1 \
    --action-feature actions \
    --max-steps  30000 \
    --batch-size 4 \
    --grad-accum 2 \
    --gpus 4  \
    --backbone "$BACKBONE" --dry-run


```

### 训练脚本-delta action

```
# 位姿 6 维转 delta，夹爪/力/力矩保持绝对
bash scripts/train_custom.sh --task autogen_mixed_charger_white --gpus 4 \
  --delta-action --backbone "$BACKBONE"

# 只转位置 3 维
bash scripts/train_custom.sh --task autogen_mixed_charger_white --gpus 4 \
  --delta-action-dims 0,1,2 --backbone "$BACKBONE"
```


### 重训练

```
BACKBONE=/workspace/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
bash scripts/train_custom.sh \
    --task autogen_mixed_flexiv_charger_white \
    --action-feature actions --camera front --camera wrist \
    --gpus 4 --backbone "$BACKBONE" \
    --save-every 2000 --keep-last 2 \
    --resume /workspace/FastWAM/runs/autogen_mixed_flexiv_charger_white/2026-10-23_04-04-09/checkpoints/state/step_001800


    --resume /workspace/FastWAM/runs/autogen_mixed_charger_white/2026-10-23_08-59-27/checkpoints/state/step_000500
```