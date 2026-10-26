# 问题记录：训练刚进第一个 step 就崩溃 —— 解码了 shape_meta 里没用的 depth 相机

## 1. 现象

run `2026-10-25_15-44-43`（`autogen_mixed_ur5`）：

| 时间 | 事件 |
|---|---|
| 15:46:02 | 开始算 dataset stats |
| 18:00:22 | rank 0/4 的 114 个 episode 算完（8059 s），进入 `all_reduce` |
| 18:04:54 | `>> Accelerate training` |
| 18:04:57 | `Train/val dataset size: 223686/223686` |
| 18:04:58~18:05:05 | 8 秒内刷出 **322** 条 `Error loading sample N (attempt k). Retrying with a random index.` |
| 18:05:05 | `RuntimeError: Failed to load a valid sample after 5 attempts for index 190310.` → rank 1 exitcode 1 → 进程组整体退出 |

即：stats 算完了，训练刚要开始取数据就挂，**一个 step 都没跑完**。

## 2. 真正原因：解码了没人用的 depth 视频

报错本身是视频时间戳超容差：

```
AssertionError: One or several query timestamps unexpectedly violate the tolerance
  (tensor([0.0667]) > tolerance_s=0.0001)
queried timestamps: tensor([566.7330])
loaded timestamps:  tensor([566.7997])
video: /workspace/data/ur5e_v30/pick_place_pens_blocks_170-merge_train_v3.0/videos/
       observation.images.wrist_depth/chunk-000/file-000.mp4
```

把 322 条报错按视频统计，**全部是 `*_depth`**：

| 数据集 | 失败的 video | 条数 |
|---|---|---|
| `pick_place_pens_blocks_170-merge_train_v3.0` | `observation.images.wrist_depth` | 204 |
| `pick_place_three_blocks_150_train_v3.0` | `observation.images.wrist_depth` | 66 |
| `pick_place_block_150_train_v3.0` | `observation.images.wrist_depth` | 49 |
| `pick_place_block_150_train_v3.0` | `observation.images.third_person_depth` | 3 |

而 `configs/data/autogen_mixed_ur5.yaml` 的 `shape_meta` 里只有 `wrist` 和 `third_person` —— **depth 相机模型根本不用**。

原因在 loader：v3 的 `_query_timestamps()` / `_query_videos()` 遍历的是 `self.meta.video_keys`（数据集里列出的**全部** video 列，这 3 个数据集各有 4 个：wrist / third_person / wrist_depth / third_person_depth），而不是 `delta_timestamps` 里真正请求的 2 个相机。v2 loader 的 `_get_query_timestamps()` 是同样的写法。于是没人用的 depth 也被逐帧解码、也被送进 `tolerance_s` 检查。

而这些 depth 流本身确实是坏的（15 fps，1 帧 = 66.7 ms）：

- 时间戳漂移 `0.0003 ~ 0.0667 s`（0.0003 s 就超过 `tolerance_s=1e-4`，0.0667 s = 整整 1 帧）；
- `pick_place_three_blocks_150_train_v3.0` 的 `observation.images.third_person_depth` 有文件**直接不存在**：`file-019.mp4` ~ `file-023.mp4`。

失败率实测（随机 200 个样本，绕开重试逻辑直接取 `multi_dataset[idx]`）：

| 结果 | 数量 |
|---|---|
| `wrist_depth` 时间戳超容差 | 164 |
| `third_person_depth` 时间戳超容差 | 1 |
| `third_person_depth` 文件不存在 | 5 |
| 成功 | 30 |

即 **85% 的样本取不出来**。`BaseLerobotDataset.__getitem__` 失败后会换随机 index 重试，最多 5 次，全失败的概率 ≈ `0.85^5 ≈ 44%`，所以训练必然在头几秒就崩（实测 64/200 直接抛 `RuntimeError`）。日志里那 322 条报错是 4 个 rank × 多个 worker 一起刷出来的。

## 3. 修复

两个 loader 都加一个 `decoded_video_keys` property，视频解码只针对 `delta_timestamps` 里请求过的列：

```python
@property
def decoded_video_keys(self) -> list[str]:
    """只解码 shape_meta 请求过的相机，而不是数据集里的全部视频列。"""
    if self.delta_indices is None:          # 没传 delta_timestamps，保持旧行为
        return list(self.meta.video_keys)
    return [key for key in self.meta.video_keys if key in self.delta_indices]
```

- `_query_timestamps()`（v3）/ `_get_query_timestamps()`（v2）改成遍历 `self.decoded_video_keys`；
- `__getitem__` 里视频解码的门控条件同样改用 `self.decoded_video_keys`；
- `image_transforms` 循环加 `if camera_key in item`，只对真正解码出来的相机做变换。

涉及文件：`src/fastwam/datasets/lerobot3/lerobot_dataset.py`、`src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py`。

效果（真实数据）：

- `ur5e_v30` 3 个数据集：修复前 200 个随机样本 170 个失败，修复后 **1000/1000 全部成功**；解码列变为 `['observation.images.wrist', 'observation.images.third_person']`；
- `guodi_sanitized_v21`（v2.1，3 个相机全用到）：`decoded_video_keys` = 全部 3 个，`40/40` 样本正常，行为不变；
- 顺带：每个样本的视频解码从 4 路降到 2 路，数据加载开销大致减半。

## 4. 数据侧建议

- 模型不用 depth，转换 v3 / 采集时就不该把 depth 流写进去：省磁盘、省转换时间，也少一类故障源。修复后即使保留也不会再报错（不再被解码）。
- 如果以后要用 depth，这两个数据集得重导：`pens_blocks_170` 的 `wrist_depth` 与其他相机**差 1 帧**；`three_blocks_150` 的 `third_person_depth` **缺 `file-019~023`**。
- **不建议靠调大 `tolerance_s` 绕过**：1 帧偏差就是 0.0667 s，调到能通过的 0.07 s 等于静默用错帧训练。

## 5. 附带：这次 run 的 `dataset_stats.json` 可以复用

stats 在崩溃之前已经算完并落盘：

```
runs/autogen_mixed_ur5/2026-10-25_15-44-43/dataset_stats.json   # 455 episodes / 223,686 帧
```

下次启动用 `--norm-stats`（`data.train.pretrained_norm_stats`）指过去，就能直接跳过 2 小时 14 分的 stats 计算。这份文件与新的「按列读 parquet」实现逐项比对，24 个统计量最大绝对误差 **`0.000e+00`**（见 `docs/prob_nccl.md` 第 8 节）。
