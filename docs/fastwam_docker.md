# FastWAM Docker：在 8 卡宿主机上创建 4 卡容器

本文只讲一件事：**用仓库构建出的 `fastwam:cu126` 镜像，在 8 卡宿主机上只给容器 4 张卡**。
镜像构建、挂载布局、容器内首次准备等完整说明见 [docker/README_zh.md](../docker/README_zh.md)。

`docker/run_container.sh` 已经预留了 `GPUS` 环境变量，会原样传给 `docker run --gpus`
（`docker/run_container.sh:24`、`:50`），因此不需要改脚本。

---

## 1. 推荐写法：显式指定 4 张卡

```bash
# 在仓库根目录执行，指定物理 GPU 0,1,2,3
GPUS='"device=0,1,2,3"' bash docker/run_container.sh
```

> **不要写成 `GPUS=4`。**
> Docker 的 `--gpus 4` 会被解析成「只使用设备号为 4 的那一张卡」，而不是「使用 4 张卡」，
> 设备号不存在时会直接报 `nvidia-container-cli: initialization error: nvml error`。

如果不在意具体是哪 4 张（同型号卡），也可以按数量取：

```bash
GPUS='"count=4"' bash docker/run_container.sh
```

`count` 按 NVML 枚举顺序取前 4 张，通常就是 `nvidia-smi` 的 0–3，但不如 `device=` 可控。

---

## 2. 等价的纯 `docker run` 命令

```bash
docker run --rm -it \
  --gpus '"device=0,1,2,3"' \
  --ipc=host --shm-size=32g --network=host \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/runs:/workspace/FastWAM/runs \
  -v $(pwd)/evaluate_results:/workspace/FastWAM/evaluate_results \
  -v $(pwd)/hf_cache:/workspace/hf_cache \
  fastwam:cu126
```

`docker/README_zh.md` 里的手动命令默认是 `--gpus all`，把这一项换掉即可。

---

## 3. 容器内验证

进入容器后只会看到这 4 张卡，且会被重新编号为 0–3：

```bash
nvidia-smi -L
python -c "import torch; print(torch.cuda.device_count())"   # 应输出 4
```

---

## 4. 启动训练 / 评测时也要改成 4

镜像里的配置默认是 8 卡，需要覆盖：

```bash
# 训练：第一个参数就是进程数
bash scripts/train_zero1.sh 4 task=libero_uncond_2cam224_1e-4

# LIBERO 评测（configs/sim_libero.yaml 默认 num_gpus: 8）
python experiments/libero/run_libero_manager.py ... MULTIRUN.num_gpus=4

# RoboTwin 评测（configs/sim_robotwin.yaml 默认 num_gpus: 8）
python experiments/robotwin/run_robotwin_manager.py ... MULTIRUN.num_gpus=4
```

`run_libero_manager.py` 会校验 `CUDA_VISIBLE_DEVICES` 的数量，填 8 会直接报错。

---

## 5. 注意事项

- **同时跑两个 4 卡容器**：用不相交的设备号，如 `device=0,1,2,3` 和 `device=4,5,6,7`，
  设备号以宿主机 `nvidia-smi` 的 index 为准。
- **卡型不一致时务必用 `device=`**，`count=4` 可能混入不同型号 / 显存不同的卡。
- **端口冲突**：容器默认 `--network=host`。多容器并存时注意训练端口（默认 29500）、
  LIBERO manager 端口，以及 WebSocket 推理服务端口（8765，见
  [serving_ws_api_zh.md](serving_ws_api_zh.md)）。
- **卡序一致性**：需要避免 CUDA 按算力重排导致卡序与 `nvidia-smi` 不一致时，可加
  `-e CUDA_DEVICE_ORDER=PCI_BUS_ID`（脚本支持通过 `EXTRA_DOCKER_ARGS` 追加）：

```bash
EXTRA_DOCKER_ARGS='-e CUDA_DEVICE_ORDER=PCI_BUS_ID' \
GPUS='"device=0,1,2,3"' bash docker/run_container.sh
```
