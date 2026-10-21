# FastWAM Docker 使用说明

面向 **x86_64 + CUDA 12.x 驱动（≥ 525，包括只支持到 CUDA 12.1 的 ~530 驱动）** 机器的
训练 / 推理 / LIBERO 评测一体化镜像。

## 镜像设计要点

- **PyTorch 2.7.1+cu126**：torch 2.7.1 没有 cu121 wheel，但 cu126 wheel 自带完整 CUDA 运行库，
  通过 CUDA minor version compatibility 可在**任何 12.x 驱动**（含 12.1 驱动）上运行。
  镜像内已把 `pyproject.toml` 的 `+cu128` 后缀去掉（只改镜像内副本，不动仓库文件）。
- **devel 基础镜像**（含 nvcc 12.6）：DeepSpeed 的 op 在首次训练时 JIT 编译需要编译器。
  首次训练会多花几分钟编译，之后缓存到 `/root/.cache/torch_extensions/`。
- **代码打进镜像**（`/workspace/FastWAM`），数据 / 权重 / 训练输出全部挂载。
- **LIBERO 评测环境**已装好（`/opt/LIBERO` + `mujoco==3.3.2`，EGL 离屏渲染）。
  LIBERO 以 `--no-deps` 方式安装，避免它的旧依赖降级 fastwam 锁定的包。

## 1. 构建镜像

在仓库根目录执行（需要外网访问 pip / deadsnakes PPA / GitHub）：

```bash
docker build -f docker/Dockerfile -t fastwam:cu126 .
```

构建结束时镜像会自动做 `pip check` + 关键包导入自检，失败即构建失败。
镜像约 15–20 GB。

## 2. 启动容器

推荐用脚本（默认挂载仓库旁的 `data/ checkpoints/ runs/ evaluate_results/ hf_cache/`）：

```bash
bash docker/run_container.sh
```

或手动：

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host --shm-size=32g \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/runs:/workspace/FastWAM/runs \
  -v $(pwd)/evaluate_results:/workspace/FastWAM/evaluate_results \
  -v $(pwd)/hf_cache:/workspace/hf_cache \
  fastwam:cu126
```

| 容器路径 | 宿主机 | 内容 |
| --- | --- | --- |
| `/workspace/FastWAM` | 不挂载（代码在镜像内） | FastWAM 代码 |
| `/workspace/data` | 挂载 | LeRobot 数据集、文本嵌入缓存 |
| `/workspace/checkpoints` | 挂载 | Wan2.2 权重（自动下载）、ActionDiT backbone、released/自训 checkpoint、dataset_stats |
| `/workspace/FastWAM/runs` | 挂载 | 训练输出（**必须挂载**，否则随容器销毁） |
| `/workspace/FastWAM/evaluate_results` | 挂载 | 评测输出 |
| `/workspace/hf_cache` | 可选 | HuggingFace 下载缓存（避免重复下载） |

> `--ipc=host --shm-size=32g`：dataloader 多 worker + accelerate 多进程需要大共享内存。
> 多机训练时容器内可用 `--network=host`（脚本已默认开启）直接复用主机网络。

## 3. 首次启动准备（容器内执行一次）

```bash
# 1) Wan2.2 权重会自动下载到 /workspace/checkpoints（DIFFSYNTH_MODEL_BASE_PATH 已设置）

# 2) 生成 ActionDiT backbone（训练前必须）
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output /workspace/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16

# 3) （可选）下载 released checkpoint 做 LIBERO 评测
huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json \
  --local-dir /workspace/checkpoints/fastwam_release
```

## 4. 训练 / 推理示例（容器内）

```bash
# 文本嵌入缓存
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4

# 训练（首次自动计算 dataset_stats.json，输出到 /workspace/FastWAM/runs/）
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4

# LIBERO 评测（released checkpoint）
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/workspace/checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=/workspace/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.sigma_shift=5.0 \
  MULTIRUN.num_gpus=8
```

自定义 LeRobot 数据的训练与推理流程见
[docs/custom_data_tutorial_zh.md](../docs/custom_data_tutorial_zh.md)，
所有路径按上表的挂载约定替换（数据放 `/workspace/data/...`）。

## 5. 快速验证

构建后先跑冒烟检查：

```bash
# CUDA / 驱动兼容性（在 12.1 驱动机器上应输出 2.7.1+cu126 True 12.6）
docker run --rm --gpus all fastwam:cu126 \
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"

# LIBERO 导入 + EGL（无显示器离屏渲染）
docker run --rm --gpus all fastwam:cu126 \
  python -c "from libero.libero.envs import OffScreenRenderEnv; print('libero egl ok')"
```

进入容器后可用 `scripts/dryrun_fastwam.py` 验证模型推理链路，
再用 `max_steps=5` 跑一次短训练确认 DeepSpeed JIT 编译与 `runs/` 挂载落盘正常。

## 6. 常见问题

| 现象 | 处理 |
| --- | --- |
| 首次训练卡在编译几分钟 | DeepSpeed JIT 在编译 op，属正常；如需加速可设 `TORCH_CUDA_ARCH_LIST`（如 `"8.9"`）后重进容器 |
| LIBERO 渲染报 EGL 错误 | 确认运行时 `--gpus` 包含 graphics 能力（镜像已设 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`；若用自定义 docker run 请勿覆盖该变量） |
| `torch.cuda.is_available()` 为 False | 宿主机未装 nvidia-container-toolkit，或驱动 < 525（CUDA 12.0 以下不支持 minor version compatibility） |
| 想换数据/权重目录 | 用 `HOST_DATA_DIR=... HOST_CKPT_DIR=... bash docker/run_container.sh` 覆盖 |
