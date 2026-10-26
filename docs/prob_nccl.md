# 问题记录：训练启动卡在 dataset stats 计算，被 NCCL watchdog 判为 collective 超时（SIGABRT）

- 发生时间：2026-10-24 04:56 启动，05:38 崩溃
- 任务：`autogen_mixed_guodi_wrc`，4 卡，DeepSpeed ZeRO-1
- 日志：`log/2026-10-24-0456_fastwam_autogen_mixed_guodi_wrc.log`
- 结论：**不是 NCCL / 网络 / 硬件 / 显存问题**。rank0 在单进程遍历 1497 个 episode 算归一化统计量，跑了 **38~40 分钟**；另外 3 个 rank 全程干等在 `broadcast_object_list` 这一个 collective 上，等待时间超过进程组默认的 **30 分钟**超时，watchdog 在 rank0 刚算完的瞬间把 4 个进程一起 abort。
- 直接解法：**复用已算好的 `dataset_stats.json`**（`--norm-stats`），跳过这段计算，启动时间从 ~39 分钟降到秒级。

---

## 1. 现象

```
[rank2]:[E1024 05:38:17 ProcessGroupNCCL.cpp:632] [Rank 2] Watchdog caught collective operation timeout:
         WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=1, NumelOut=1, Timeout(ms)=1800000)
         ran for 2337512 milliseconds before timing out.
[rank2]:[E1024 05:38:17 ProcessGroupNCCL.cpp:2271] failure detected by watchdog at work sequence id: 1
         PG status: last enqueued work: 1, last completed work: -1
...
E1024 05:38:19 torch/distributed/elastic/.../api.py:874] failed (exitcode: -6) local_rank: 2 (pid: 868)
ChildFailedError: scripts/train.py FAILED
Root Cause: rank 2, exitcode -6 (SIGABRT)
```

几个关键字段：

| 字段 | 值 | 含义 |
|---|---|---|
| `SeqNum=1` | 第 1 个通信操作 | 训练还没真正开始 |
| `OpType=BROADCAST, Numel=1` | 广播 1 个元素 | `broadcast_object_list` 的第一步：先广播对象的字节数（一个 int64），见 `robot_video_dataset.py:111` |
| `last completed work: -1` | 一个都没完成 | 通信器从头到尾没跑通过 |
| `Timeout(ms)=1800000` | 30 分钟 | 进程组默认超时 |
| `ran for 2337 s` | 排队 39 分钟 | 回溯到 04:59:20 |

`2337 s` 这个数字就是本案的钥匙：05:38:17 往前推 39 分钟 = **04:59:20**，正好是 rank0 打印 `>> Calculating dataset stats for normalization...` 的那一秒。

## 2. 真正卡住的位置

| 时间 | 事件 |
|---|---|
| 04:56:50 | accelerate 启动 4 个 rank |
| 04:59:00 | Wan2.2 组件加载完（116.66 s） |
| 04:59:12 | ActionDiT backbone 加载完（`keys=820`） |
| **04:59:20** | rank0 打印 `Calculating dataset stats`（`robot_video_dataset.py:103`）并开始逐个 episode 遍历；**同一秒**另外 3 个 rank 进入 `broadcast_object_list` 入队 —— 就是 watchdog 说的那个 `SeqNum=1 BROADCAST` |
| **≈05:38:10** | `dataset_stats.json` 落盘（文件 mtime 换算到容器时间），即统计量算完，耗时约 **2330 s** |
| 05:38:17 | watchdog 判定该 collective 已排队 2337 s > 1800 s，abort；rank 2/3 退出码 -6 |

代码位置：

```python
# src/fastwam/datasets/lerobot/robot_video_dataset.py:99
if not pretrained_norm_stats:
    if not is_training_set:
        raise ValueError("pretrained_norm_stats must be provided for validation/test sets ...")
    if PartialState().is_main_process:
        logger.info("Calculating dataset stats for normalization...")   # 04:59:20
        dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)   # ← 39 分钟都在这
        work_dir = misc.get_work_dir()
        save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
    else:
        dataset_stats = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        obj_list = [dataset_stats]
        torch.distributed.broadcast_object_list(obj_list, src=0)   # ← 其余 rank 卡在这里
        dataset_stats = obj_list[0]
```

**只有 rank0 在算，其余 rank 全部在广播里等**。而 `get_dataset_stats`（`base_lerobot_dataset.py:325-355`）是逐 episode 的 Python 循环（ThreadPoolExecutor，但受 GIL 限制基本是串行），耗时与 episode 数成正比：

- 10-23 的 `autogen_mixed_charger_white`：527 episodes / 108,468 帧 → 统计只要 **5 分半**（09:26:56→09:32:46 用时 350 s）
- 10-24 的 `autogen_mixed_guodi_wrc`：**1497 episodes / 971,872 帧**（episode 数 2.8×、帧数 9×）→ 实测 **~39 分钟**

旁证：06:28 那次重跑的日志里有进度条，`147/1497 [03:57]` → 约 1.6 s/episode → 全量外推 ≈ 40 分钟，与上面的 39 分钟吻合。

排除项：

- ❌ NCCL / 网络 / NVLink：06:28 重跑开了 `NCCL_DEBUG=INFO`，通信器正常建立（`NCCL 2.26.2`、`Channel 0..23 ... via P2P/CUMEM`），没有报错。卡住的这个 collective 是**应用层自己发的 stats 广播**，不是训练通信
- ❌ 显存 / OOM：显存不足会直接抛 `CUDA out of memory`
- ❌ 抢卡：统计循环是纯 CPU（读 parquet），与 GPU 无关；04:56 那次预检也通过了
- ❌ 代码 bug：10-23 三次同样流程都正常，只是数据集体量小，统计 5 分半 < 30 分钟超时
- ✅ 本质：**"rank0 单干 + 其它 rank 等一个 collective" 的结构，撞上了 30 分钟的进程组超时**

## 3. 直接解法：复用历史 `dataset_stats.json`

**好消息：这份 stats 其实已经算完了。** 就在崩溃那次 run 的目录里：

```
# 宿主
/mnt/sfs_turbo/h00672373/docker_dir/runs/autogen_mixed_guodi_wrc/2026-10-24_04-56-54/dataset_stats.json
# 容器内
/workspace/FastWAM/runs/autogen_mixed_guodi_wrc/2026-10-24_04-56-54/dataset_stats.json
```

内容校验过：`num_episodes=1497`、`num_transition=971872`、action/state 各 12 组统计量齐全、`stepwise_*` 形状为 (32, 24)/(1, 24)，是完整的全量统计。

复用方式（`train_custom.sh` 早就支持，`scripts/train_custom.sh:525` → `data.train.pretrained_norm_stats`）：

```bash
STATS=/workspace/FastWAM/runs/autogen_mixed_guodi_wrc/2026-10-24_04-56-54/dataset_stats.json

bash scripts/train_custom.sh \
  --task autogen_mixed_guodi_wrc \
  --action-feature action \
  --gpus 4 --batch-size 8 --grad-accum 2 --max-steps 20000 \
  --backbone /workspace/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --norm-stats "$STATS"
```

直接用 `train_zero1.sh` 时等价写法：

```bash
bash scripts/train_zero1.sh 4 task=autogen_mixed_guodi_wrc \
  batch_size=8 max_steps=20000 gradient_accumulation_steps=2 \
  data.train.pretrained_norm_stats="$STATS" \
  model.action_dit_pretrained_path=/workspace/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

效果：

- 4 个 rank 都不再遍历数据集，直接读 145 KB 的 json（`robot_video_dataset.py:114`），启动从 ~39 分钟降到秒级
- 不会再撞 30 分钟 NCCL 超时
- 每次运行仍会把当时用的 stats 复制一份到 `runs/<task>/<run_id>/dataset_stats.json`（`robot_video_dataset.py:118`），所以**任何一个历史 run 目录里都有一份可直接复用的**

建议长期做法：把这份 json 放到固定位置（例如 `/workspace/data/guodi_sanitized_v30/dataset_stats.json`，或仓库内 `configs/data_stats/`），并写进 `configs/data/autogen_mixed_guodi_wrc.yaml` 的 `pretrained_norm_stats`，以后不用每次带 `--norm-stats`。注意该 config 是 `prepare_custom_training.py` 自动生成的，用 `--dataset` 重新生成时会被重置成 `null`。

## 4. 复用前必须确认：stats 的"口径"没变

stats 是在 `preprocessor.action_state_transform(batch)` **之后**统计的（`base_lerobot_dataset.py:299`），所以下列任何一项变了都必须重算，不能复用：

| 变化 | 为什么 |
|---|---|
| `dataset_dirs` 增删数据集 | 样本集变了 |
| `shape_meta` 里 state/action 的 key、维度、`lerobot_key` | 统计的列变了 |
| `--delta-action` / `delta_action_dims` / `delta_action_base` | 动作先被转成 delta 再统计，复用绝对动作的 stats 会**把归一化尺度搞错** |
| `num_frames` | 动作滑窗长度变了，`stepwise_*` 的 T 维对不上 |

json 里只有 `num_episodes` / `num_transition` 两个计数，可以据此做粗校验，但没有自动指纹（哈希）校验，建议缓存文件名带上数据集版本或日期。

## 5. 长期改进（按性价比）

1. **优先复用**（零代码改动）：`--norm-stats` / `data.train.pretrained_norm_stats`。
2. ✅ **已实现：统计计算分片到所有 rank + `all_reduce`**（见第 7 节）。4 卡并行，实测 **2330 s → 626 s（3.7×）**。
3. **不要把 stats 用 `broadcast_object_list` 传**：✅ 已随第 2 条一起去掉——分片后每个 rank 通过 `all_reduce` 直接拿到相同结果，不再需要广播。
4. ✅ **已实现：统计循环按列读 parquet**（见第 8 节）。`get_episode_data` 原来每个 episode 都 `hf_dataset.select()` 一次、再把该 episode 的**所有非视频列**转成 Python list（`lerobot3/lerobot_dataset.py:260-273`），而 stats 只需要 action / state 两列；现在改成每个数据集一次 `pyarrow.parquet.read_table(columns=[...])`，实测比逐 episode 读快 2~3 个量级。
5. **进程组超时**：1800 s 是 torch 默认值，这类"单 rank 长计算 + 集体等待"的结构天然容易撞上。分片后单 rank 的计算量降到 1/N，风险大幅下降，但 episode 数特别大时仍可考虑调长 timeout。

## 6. 附：几次运行的统计耗时对照

| run | 数据集 | episodes / 帧数 | stats 计算耗时 |
|---|---|---|---|
| 2026-10-23 09:25 | `autogen_mixed_charger_white` | 527 / 108,468 | 350 s（09:26:56→09:32:46） |
| 2026-10-23 11:40 | `autogen_mixed_charger_white` | 527 / 108,468 | 337 s（11:41:38→11:47:15） |
| 2026-10-24 04:56 | `autogen_mixed_guodi_wrc` | 1497 / 971,872 | ~2330 s，超时崩溃 |
| 2026-10-24 06:28 | `autogen_mixed_guodi_wrc` | 1497 / 971,872 | tqdm 外推 ~40 min（未跑完） |
| 2026-10-24 07:27 | `autogen_mixed_guodi_wrc` | 1497 / 971,872 | **626 s**（4 rank 分片，07:28:56→07:39:22） |

结论：**30 分钟的 NCCL 超时对"中等以上规模数据集"的 stats 计算是随时会踩的坑**。分片（第 7 节）已把单次耗时压到安全区，缓存 stats 则是连这 10 分钟也省掉的做法。

## 7. 分片计算 stats（已实现）

改动前：只有 rank0 遍历全部 episode，其余 rank 在 `broadcast_object_list` 里干等（本次崩溃的直接原因）。
改动后：episode 按 `range(rank, N, world_size)` 轮转分给每个 rank，各自算局部聚合，再用 3 次 `all_reduce` 合并，**每个 rank 都直接得到最终结果，广播被彻底去掉**。

涉及文件：

- `src/fastwam/datasets/lerobot/base_lerobot_dataset.py`
  - `get_dataset_stats(preprocessor, rank=0, world_size=1)`：按 rank 分片遍历；每个 episode 仍算 min / max / q01 / q99 / mean / var，但 mean / var 以 **float64 累加求和**（`Σmean`、`Σvar`、`Σmean²`）参与合并。
  - `_combine_sharded_stats()`：把各 rank 的局部结果拼成一维 float64 tensor，分别用 `MIN` / `MAX` / `SUM` 三次 `all_reduce` 合并，再拆回原来的形状。
  - `_stats_all_reduce_device()`：NCCL 后端用 `cuda:LOCAL_RANK`，gloo 用 CPU。
- `src/fastwam/datasets/lerobot/robot_video_dataset.py`
  - `_stats_shard_info()`：取 `(rank, world_size)`（`PartialState()` 负责拉起进程组）。
  - 计算后各 rank 结果一致，仅 rank0 落盘 `dataset_stats.json`。

`mean / std` 的合并公式（与原来的两遍式完全等价，已用数值测试验证）：

```
stepwise_mean = Σmean / N
stepwise_std  = sqrt(Σvar/N + Σmean²/N - stepwise_mean²)
global_mean   = mean_t(Σmean) / N
global_std    = sqrt(mean_t(Σvar)/N + mean_t(Σmean²)/N - global_mean²)
```

几点说明：

- 分片是**结果等价**的，不是近似：min / max / 分位数本来就是"跨 episode 取极值"，mean / std 是"每个 episode 等权"，求和后除以总 episode 数即可。
- 退化保护：`world_size > episode 数` 时自动退回单进程全量计算（每个 rank 各算一遍，结果仍一致）。
- 关闭开关：`FASTWAM_STATS_SHARDING=0`，退回"每个 rank 全量计算"（不再用 `broadcast_object_list`，也就不会再有那个 39 分钟的超时）。
- 日志：每个 rank 都会打印 `rank r/W: computing stats over M of N episodes` 与 `rank r/W: local episodes done in X s`，便于确认各 rank 都在干活（tqdm 进度条仍只在 rank0 显示，避免多进程输出互相覆盖）。
- 验证（离线）：以 1 / 2 / 4 rank 三种配置（gloo，多进程）对同一份合成数据集计算，与原始单进程实现的最大绝对误差为 `2.4e-07`（float32 精度量级），退化路径同样一致。
- 验证（真实数据，2026-10-24 07:27 run）：1497 episodes → `375/375 [10:26]`，总计 **626 s**（原 ~2330 s，3.7×）；`all_reduce` 在 **NCCL + CUDA float64** 上正常完成，随后 `>> Accelerate training` 正常启动，全程无超时。与该 run 之前 04:56 那次单进程算出的 stats 逐项比对，26 个叶子统计量**最大绝对误差 1.19e-07**，即分片结果与原实现等价。

## 8. 按列读 parquet 计算 stats（已实现）

第 7 节解决了"只有 rank0 干活"的问题，但每个 episode 的读取成本还在。`get_episode_data` 走的是训练用的通用读取路径：`hf_dataset.select(range(start, end))` 之后对这个 episode 的**每一列**做 Arrow → numpy → tensor，最后再按 `delta_timestamps` 取滑窗。对 `ur5e_v30` 这类带相机内外参、力/力矩等 26 个非视频列的数据集，stats 只需要其中 2 列，却要为另外 24 列付出 15 倍以上的读取与转换开销（warm cache 实测 0.250 s/ep vs 0.014 s/ep）。

### 设计

- `src/fastwam/datasets/lerobot/utils/parquet_columns.py`（新增）
  - `check_bulk_columns(dataset, keys)`：**开跑前**把整份布局验一遍——`pq.read_schema` 查列，再单读一次 `episode_index` 确认每个 episode 都是连续块且都在文件里；不满足就抛 `UnsupportedBulkRead`。这样正式读取不会中途失败、留下半份统计量。（单读 `episode_index` 实测 0.3 s 量级，且顺带预热了页脚，后面那次全量读更快。）
  - `iter_bulk_episodes(dataset, keys)`：对每个数据集只跑**一次** `pq.read_table(paths, columns=[*keys, "episode_index"])`，按 `episode_index` 的连续块切出每个 episode 的 `{列名: 行}`，逐 episode yield。
  - `iter_multi_dataset_episodes(datasets, keys)`：多个数据集按顺序拼接，episode 编号转成全局编号。
  - `_column_to_numpy()`：v3 的 `fixed_size_list<float, D>` 与 v2 的 `list<float>` 都还原成 `[T, D]`；行宽不一致或 episode 行不连续时抛 `UnsupportedBulkRead`，由调用方回退。
- `src/fastwam/datasets/lerobot/base_lerobot_dataset.py`
  - `_accumulate_stats_from_columns()`：stats 循环优先走这条路径。仍然复用 `_assemble_episode_sample()` + `preprocessor.action_state_transform()`，所以**统计口径（滑窗、delta action、维度）与训练时完全一致**，只是数据来源从"逐 episode 读全列"换成"一次读需要的列"。
  - 分片逻辑不变：先按 `range(rank, N, world_size)` 过滤，再只累加属于本 rank 的 episode。
  - 任一数据集不支持（缺列、缺 `episode_index`、行宽不一致、数据集类没实现读取方法）时打印一行日志并**回退到原来的逐 episode 路径**，行为不受影响。
- `lerobot3/lerobot_dataset.py` / `lerobot/lerobot_dataset.py`：`LeRobotDataset` 提供 `bulk_data_paths()` / `bulk_episode_ids()`，`MultiLeRobotDataset` 提供 `check_bulk_columns()` / `iter_bulk_episode_columns()`。两种布局都能读：v3 是 `data/*/*.parquet`，v2 是 `meta.get_data_file_path(ep)` 的每 episode 一个文件。

开关：`FASTWAM_STATS_BULK_READ=0` 强制走旧的逐 episode 路径（用于对比或排障）。

### 实测

| 数据集 | episodes / 帧数 | 逐 episode | 按列读 | 加速 |
|---|---|---|---|---|
| `ur5e_v30` 混合（26 个非视频列） | 455 / 223,686 | 5.4 s/ep → **40.9 min** | **2.1 s** | ~1000× |
| `guodi_sanitized_v21` 混合（7 列，v2.1） | 1497 / 971,872 | 52.3 s（线程池） | **10.5 s** | 5× |

`ur5e_v30` 的 5.4 s/ep 是本机离线测得的；容器里跑训练那次（2026-10-25 15:44）是 5~90 s/ep，主要被 SFS 上的随机读拖慢，按列读把这部分几乎完全消掉。

### 正确性验证

- **样本一致**：3 个 `ur5e_v30` 数据集跨边界取 9 个 episode、`guodi_sanitized_v21` 取 9 个 episode，按列读拼出的 `{action, state}` 与 `get_episode_data()` 的结果**逐位相同**（max abs diff `0.000e+00`）。
- **统计一致**：`guodi_sanitized_v21` 上按列读与 `FASTWAM_STATS_BULK_READ=0` 的 24 个统计量**完全相等**（`0.000e+00`），`num_episodes` / `num_transition` 不变。
- **分片一致**：2 rank（gloo）与单进程结果同样 `0.000e+00`。
- **端到端**：与 `2026-10-25_15-44-43` 那次 run（旧代码、4 rank、NCCL 分片）在真实数据集上算出的 `dataset_stats.json` 逐项比对，24 个统计量**最大绝对误差 `0.000e+00`**。
- **对着独立参考**：另写一份"直接从 parquet 列算 stats"的实现（复用同样的 float32 归约 + float64 累加），在 `ur5e_v30` 的 16 个统计量上最大误差 `1.19e-07`（float32 量级），`num_episodes=455` / `num_transition=223686` 不变。
- 精度旁注：若参考实现改用纯 numpy float64（`np.var(dtype=float64)`），`state.stepwise_std` 会差 `7.1e-04`（相对 `1.4e-3`）。这一项来自**原有**实现里 `torch.var` 先在 float32 上算方差（`Σvar` 之后才升 float64），已单独验证把参考也换成 float32 求方差后差值归零，属于既有精度特性，本次没有改动（如需更准可把 `accumulate()` 里的 `cur.var(0)` 改成 float64，代价是与历史 stats 有一处 `1e-3` 量级的变化）。

