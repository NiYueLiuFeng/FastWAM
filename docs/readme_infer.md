# FastWAM 推理服务教程（autogen_mixed_charger_white）

本文是容器内创建 FastWAM 推理服务的完整操作教程，使用训练后权重：

```text
/mnt/sfs_turbo/h00672373/docker_dir/runs/autogen_mixed_charger_white/2026-10-23_11-40-22
```

这个仓库已经内置了 WebSocket 推理服务（`scripts/serve.py` + `src/fastwam/serving`），
这个权重也有配套的 task 配置（`configs/task/autogen_mixed_charger_white.yaml`），
不需要自己写服务端。

**这套权重的关键参数**

| 项 | 值 |
|---|---|
| task | `autogen_mixed_charger_white` |
| checkpoint | `.../checkpoints/weights/step_020000.pt`（12 GB） |
| 归一化统计 | `.../dataset_stats.json` |
| 相机 | `front`、`wrist`，各 `[3,224,224]`，横向拼成 `224x448` |
| state / action | 各 13 维 |
| action horizon | 32 步 |
| 输入 | 2 张原始 JPEG/PNG + 13 维原始 state + 任务指令 |
| 输出 | `[32,13]` 的 action chunk |
| 服务 | `ws://<host>:8765/v1/ws` |

> 说明：当前 shell 没有 docker 权限，无法替读者实际启动容器；下面的路径和接口是按仓库代码
> 逐项核对过的。启动后用 `/readyz`、`/v1/info` 和一次真实请求即可确认。

---

## 目录

- [0. 宿主机准备](#0-宿主机准备)
- [1. 启动容器](#1-启动容器)
- [2. 容器内自检](#2-容器内自检)
- [3. 启动推理服务](#3-启动推理服务)
- [4. 验证服务](#4-验证服务)
- [5. 机器人侧完整客户端](#5-机器人侧完整客户端)
- [6. 后台常驻与停止](#6-后台常驻与停止)
- [7. 常见问题](#7-常见问题)
- [8. 最短路径总结](#8-最短路径总结)

---

## 0. 宿主机准备

先确认权重和空闲 GPU：

```bash
RUN=/mnt/sfs_turbo/h00672373/docker_dir/runs/autogen_mixed_charger_white/2026-10-23_11-40-22
ls -lh "$RUN/checkpoints/weights/step_020000.pt" "$RUN/dataset_stats.json"
# 12G 的 .pt 和 ~84K 的 json

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

选一张空闲卡（下面示例用 0 号卡）。

---

## 1. 启动容器

### 1.2 启动命令

```bash
cd /mnt/sfs_turbo/h00672373/github/FastWAM

HOST_DATA_DIR=/mnt/sfs_turbo/h00672373/train_data \
HOST_CKPT_DIR=/mnt/sfs_turbo/h00672373/docker_dir/checkpoints \
HOST_RUNS_DIR=/mnt/sfs_turbo/h00672373/docker_dir/runs \
HOST_EVAL_DIR=/mnt/sfs_turbo/h00672373/docker_dir/evaluate_results \
HOST_HF_CACHE=/mnt/sfs_turbo/h00672373/docker_dir/hf_cache \
HOST_HOME_DIR=/mnt/sfs_turbo/h00672373/docker_dir/container_home_serve \
GPUS='"device=0"' \
NAME=serve \
EXTRA_DOCKER_ARGS='-v /mnt/sfs_turbo/h00672373/github/FastWAM:/workspace/FastWAM' \
bash /mnt/sfs_turbo/h00672373/code/docker/run_container.sh
```

几点说明：

- `HOST_RUNS_DIR` 必须指向 `docker_dir/runs`，否则容器里看不到 `step_020000.pt`。
- `HOST_DATA_DIR` 用 `train_data` 只是为了下面从训练数据里抽两张真实图做测试；
  **服务本身不读数据集**。
- `GPUS='"device=0"'` 只暴露 0 号卡；如果 0 号卡被占用，改成别的编号。
- 容器名会是 `fastwam-serve`。如果已存在同名容器，先 `docker rm -f fastwam-serve`。
- `--network=host` 已由脚本设置，所以宿主机直接访问 `127.0.0.1:8765`。

---

## 2. 容器内自检

进入容器后：

```bash
cd /workspace/FastWAM

python -c "import torch, fastwam; print(torch.__version__, torch.cuda.is_available()); print(fastwam.__file__)"
```

应看到类似：

```text
2.7.1+cu126 True
/workspace/FastWAM/src/fastwam/__init__.py
```

如果 `fastwam.__file__` 不是这个路径，说明仓库挂载没生效，回到第 1 步检查
`EXTRA_DOCKER_ARGS`。

确认文件：

```bash
ls -l configs/task/autogen_mixed_charger_white.yaml

RUN_DIR=/workspace/FastWAM/runs/autogen_mixed_charger_white/2026-10-23_11-40-22
ls -lh "$RUN_DIR/checkpoints/weights/step_020000.pt" "$RUN_DIR/dataset_stats.json"
```

确认 hydra 解析出来的配置正确（不会加载模型）：

```bash
python scripts/serve.py task=autogen_mixed_charger_white --cfg job --resolve \
  | grep -nE 'video_size|concat_multi_camera|num_frames|action_output_dim|proprio_output_dim|load_text_encoder|skip_dit_load_from_pretrain|action_dit_pretrained_path'
```

期望看到：

```text
video_size: [224, 448]
concat_multi_camera: horizontal
num_frames: 33
action_output_dim: 13
proprio_output_dim: 13
load_text_encoder: true
skip_dit_load_from_pretrain: true
action_dit_pretrained_path: null
```

---

## 3. 启动推理服务

在**第一个终端**（容器内）前台启动，方便看日志：

```bash
cd /workspace/FastWAM

export TASK=autogen_mixed_charger_white
export CKPT=/workspace/FastWAM/runs/autogen_mixed_charger_white/2026-10-23_11-40-22/checkpoints/weights/step_020000.pt
export DATASET_STATS=/workspace/FastWAM/runs/autogen_mixed_charger_white/2026-10-23_11-40-22/dataset_stats.json

export GPU=0
export PORT=8765
export STEPS=10
export TEXT_ENCODER_DEVICE=cuda
export COMPILE=true
export TOKEN=fastwam-dev-token

bash scripts/serve_deploy.sh
```

启动过程大致是：

1. 加载 Wan2.2 的 VAE + T5 文本编码器（约 1–2 分钟）
2. 构建模型、加载 12 GB checkpoint（约 1 分钟）
3. `torch.compile` + warmup（约 1–5 分钟）
4. 日志出现 `Service is ready` 和 `Uvicorn running on http://0.0.0.0:8765`

> 第一次启动慢是正常的，编译缓存会写到 `/workspace/home/.torchinductor`，下次重启会快很多。

### 可选参数

| 变量 | 含义 |
|---|---|
| `STEPS=10` | 去噪步数，和训练时的 `eval_num_inference_steps` 一致 |
| `TEXT_ENCODER_DEVICE=cpu` | 显存紧张时把 T5 放 CPU，省约 10 GB |
| `COMPILE=false` | `torch.compile` 失败或想跳过编译时用 |
| `REQUEST_TIMEOUT_MS=10000` | 首次请求或高负载时放宽超时 |
| `SIGMA_SHIFT=null` | 默认即可；这个 checkpoint 的 action scheduler infer shift 是 1.0 |
| `ACTION_HORIZON=null` | 默认 32，一般不要改 |

---

## 4. 验证服务

在**第二个终端**：

```bash
docker exec -it fastwam-serve bash
cd /workspace/FastWAM
```

### 4.1 健康检查

```bash
curl -s http://127.0.0.1:8765/healthz; echo
curl -s http://127.0.0.1:8765/readyz; echo
curl -s http://127.0.0.1:8765/v1/info | python -m json.tool
```

`/readyz` 返回 `{"ready":true}` 才算就绪。如果一直是 503，看第一个终端的日志；
`/readyz` 的 `error` 字段也会给出异常摘要。

`/v1/info` 应包含：

```json
{
  "model": {
    "variant": "FastWAM",
    "action_dim": 13,
    "state_dim": 13,
    "action_horizon": 32,
    "video_size": [224, 448],
    "concat_multi_camera": "horizontal",
    "cameras": [
      {"key": "front", "shape": [3, 224, 224]},
      {"key": "wrist", "shape": [3, 224, 224]}
    ],
    "action_keys": ["default"],
    "state_keys": ["default"],
    "normalization": "min/max",
    "uses_proprio": true,
    "dry_run": false
  }
}
```

### 4.2 合成图测试（会真正跑完整模型）

```bash
python examples/ws_client.py \
  --url ws://127.0.0.1:8765/v1/ws \
  --token fastwam-dev-token \
  --synthetic \
  --prompt "Plug the charger into the leftmost socket of the power outlet." \
  --repeat 3 --verbose
```

期望看到 `actions=(32, 13)`、`timing` 里有 `infer_ms`，以及 p50/p95。合成图只是纯色块，
动作没有物理意义，但这条命令能证明 **VAE 编码、文本编码、MoT 去噪、反归一化** 整条链路
是通的。

### 4.3 真实观测测试

从训练数据里抽第一帧和对应 state：

```bash
DS=/workspace/data/hs_data_v30/lerobot_charger_0324
mkdir -p /tmp/fastwam_obs

ffmpeg -y -loglevel error -ss 0 \
  -i "$DS/videos/observation.images.front/chunk-000/file-000.mp4" \
  -frames:v 1 /tmp/fastwam_obs/front.jpg

ffmpeg -y -loglevel error -ss 0 \
  -i "$DS/videos/observation.images.wrist/chunk-000/file-000.mp4" \
  -frames:v 1 /tmp/fastwam_obs/wrist.jpg

STATE=$(python - <<'PY'
import pandas as pd
df = pd.read_parquet(
    '/workspace/data/hs_data_v30/lerobot_charger_0324/data/chunk-000/file-000.parquet',
    columns=['observation.state'],
)
print(','.join(f'{x:.8f}' for x in df.iloc[0]['observation.state']))
PY
)
echo "$STATE"
```

然后发请求（注意图片顺序是 **front、wrist**，和 `hello_ack.model.cameras` 一致）：

```bash
python examples/ws_client.py \
  --url ws://127.0.0.1:8765/v1/ws \
  --token fastwam-dev-token \
  --image /tmp/fastwam_obs/front.jpg \
  --image /tmp/fastwam_obs/wrist.jpg \
  --state "$STATE" \
  --prompt "Plug the charger into the leftmost socket of the power outlet." \
  --output /tmp/fastwam_obs/actions.json --verbose
```

这个模型混合训练了两个任务，另一个任务（whiteboard 数据集）的指令是：

```text
wipe the charactor 'T' on the whiteboard at the end of 'CloudRoboT'.
```

prompt 必须和当前场景对应，否则模型会按错误的指令生成动作。

### 4.4 压测（可选）

```bash
python examples/ws_client.py \
  --url ws://127.0.0.1:8765/v1/ws \
  --token fastwam-dev-token \
  --synthetic \
  --prompt "Plug the charger into the leftmost socket of the power outlet." \
  --repeat 20
```

看 `p50/p95` 和 `timing.queue_ms`。单卡单 worker 是串行推理，多个连接会排队。

---

## 5. 机器人侧完整客户端

`examples/ws_client.py` 是一次性请求的完整实现。机器人闭环可以按下面这个模板接
（把 `robot.get_observation()` / `robot.step()` 换成你的 HSR602 API）：

```python
import asyncio
import io

import msgpack
import numpy as np
from PIL import Image
from websockets.asyncio.client import connect

URL = "ws://127.0.0.1:8765/v1/ws"
TOKEN = "fastwam-dev-token"
PROMPT = "Plug the charger into the leftmost socket of the power outlet."
REPLAN_STEPS = 5          # 每次预测 32 步，先执行前 5 步再重规划
NUM_STEPS = 10            # 去噪步数


def pack(obj):
    return msgpack.packb(obj, use_bin_type=True)


def unpack(data):
    return msgpack.unpackb(data, raw=False, strict_map_key=False)


def to_jpeg(image: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


async def run(robot):
    async with connect(URL, max_size=64 * 1024 * 1024, ping_interval=None) as ws:
        # 1) 握手：把 prompt 放到 hello 里，服务端会缓存文本 embedding
        await ws.send(pack({
            "v": 1, "type": "hello",
            "token": TOKEN,
            "client": "hsr602",
            "prompt": PROMPT,
        }))
        ack = unpack(await ws.recv())
        if ack.get("type") != "hello_ack":
            raise RuntimeError(f"handshake failed: {ack}")

        cameras = ack["model"]["cameras"]       # [front, wrist]
        request_id = 0

        while True:
            # 2) 取观测：原始 PIL 图 + 原始 13 维 state，不需要自己 resize
            obs = robot.get_observation()
            images = [
                {"key": cam["key"], "format": "jpeg",
                 "data": to_jpeg(obs[cam["key"]])}
                for cam in cameras
            ]

            req = {
                "v": 1, "type": "infer", "id": f"req-{request_id}",
                "images": images,
                "state": [float(x) for x in obs["state"]],
                "options": {"num_inference_steps": NUM_STEPS},
            }
            await ws.send(pack(req))

            # 3) 收响应（同一条连接上串行：发一个 infer、收一个 actions）
            while True:
                resp = unpack(await ws.recv())
                if resp.get("type") == "actions":
                    break
                if resp.get("type") == "error":
                    raise RuntimeError(
                        f"{resp['code']}: {resp['message']} {resp.get('detail') or ''}"
                    )
                # pong / reset_ack 忽略

            actions = np.asarray(resp["actions"], dtype=np.float32)  # [32,13]
            assert actions.shape == (ack["model"]["action_horizon"],
                                     ack["model"]["action_dim"])
            assert np.isfinite(actions).all()

            # 4) 执行前 REPLAN_STEPS 步，然后重新取观测、重新推理
            for action in actions[:REPLAN_STEPS]:
                robot.step(action)
            request_id += 1


if __name__ == "__main__":
    asyncio.run(run(MyRobot()))
```

动作语义（很重要）：

- 输出是**数据集 action 空间**的 13 维向量，顺序就是 `meta/info.json` 里的：
  `position_x/y/z, rotation_a/b/c, gripper_width, force_x/y/z, torque_xy/yz/zx`
- 这份配置的 `delta_action_dim_mask` 全为 `false`、`action_state_transforms: null`，
  所以服务端**不做 delta ↔ absolute 转换**，返回的就是数据集中记录的动作语义
- 服务端**不做**平滑、限幅、插值、时间对齐，这些要在机器人侧做
- 输入 state 必须是**原始未归一化**的 13 维，服务端会按 `dataset_stats.json` 归一化

---

## 6. 后台常驻与停止

调试好后改成后台运行（脚本会等 `/readyz` 就绪再返回）：

```bash
mkdir -p /workspace/FastWAM/runs/serve_logs

TASK=autogen_mixed_charger_white \
CKPT=/workspace/FastWAM/runs/autogen_mixed_charger_white/2026-10-23_11-40-22/checkpoints/weights/step_020000.pt \
DATASET_STATS=/workspace/FastWAM/runs/autogen_mixed_charger_white/2026-10-23_11-40-22/dataset_stats.json \
GPU=0 PORT=8765 STEPS=10 \
TOKEN=fastwam-dev-token \
BACKGROUND=1 WAIT_READY=1 WAIT_TIMEOUT=1800 \
LOG_FILE=/workspace/FastWAM/runs/serve_logs/charger_white.log \
bash scripts/serve_deploy.sh
```

查看日志 / 停止：

```bash
tail -f /workspace/FastWAM/runs/serve_logs/charger_white.log
kill "$(cat /workspace/FastWAM/runs/serve_logs/charger_white.pid)"
```

注意：后台进程是容器内 `nohup` 起来的；如果你 `exit` 掉容器的主 shell，容器停止，
服务也会停止。需要长期跑就把容器保持在后台，或把启动命令作为容器的启动命令。

---

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| `找不到任务配置 configs/task/autogen_mixed_charger_white.yaml` | 镜像太旧，仓库没挂进去。确认 `EXTRA_DOCKER_ARGS` 里有 `-v .../github/FastWAM:/workspace/FastWAM`，或重建镜像 |
| `unexpected keyword argument 'delta_action_dims'` | 同上，容器里旧的 processor 配上了新 data config；挂载当前仓库 |
| `fastwam.__file__` 不是 `/workspace/FastWAM/src/...` | 挂载没生效，检查 `EXTRA_DOCKER_ARGS` |
| `/readyz` 一直 503 | 看日志：权重路径、显存、`torch.compile`。`/readyz` 的 `error` 字段有摘要 |
| `text encoder is not loaded` | 确认 `serving.force_serve_overrides=true`（默认），或显式 `model.load_text_encoder=true` |
| 显存不足 | `TEXT_ENCODER_DEVICE=cpu`；或 `COMPILE=false`；或换空闲卡 |
| `CUDA error: no kernel image is available` | 这张卡架构不在 torch 编译范围内；H800 是 sm_90，一般不会遇到 |
| `BAD_REQUEST: expected 2 images` | 图片数量/顺序不对，顺序是 `front`、`wrist` |
| `BAD_REQUEST: state must have 13 values` | state 维度不对，必须是 13 维原始值 |
| 动作尺度明显不对 | 检查 `dataset_stats.json` 是否和 checkpoint 同一次 run；检查 prompt 是否对应任务；`SIGMA_SHIFT` 保持默认 |
| `TIMEOUT` / `BUSY` | 单卡单 worker 串行；提高 `REQUEST_TIMEOUT_MS`，或减少并发/步数 |
| 端口冲突 | `PORT=18765`，客户端 URL 同步改 |
| `torch.compile` 太慢或失败 | `COMPILE=false` 启动，推理会慢一些但更稳 |

---

## 8. 最短路径总结

```bash
# 宿主机
cd /mnt/sfs_turbo/h00672373/github/FastWAM
HOST_RUNS_DIR=/mnt/sfs_turbo/h00672373/docker_dir/runs \
HOST_DATA_DIR=/mnt/sfs_turbo/h00672373/train_data \
HOST_CKPT_DIR=/mnt/sfs_turbo/h00672373/docker_dir/checkpoints \
HOST_EVAL_DIR=/mnt/sfs_turbo/h00672373/docker_dir/evaluate_results \
HOST_HF_CACHE=/mnt/sfs_turbo/h00672373/docker_dir/hf_cache \
HOST_HOME_DIR=/mnt/sfs_turbo/h00672373/docker_dir/container_home_serve \
GPUS='"device=0"' NAME=serve \
EXTRA_DOCKER_ARGS='-v /mnt/sfs_turbo/h00672373/github/FastWAM:/workspace/FastWAM' \
bash /mnt/sfs_turbo/h00672373/code/docker/run_container.sh

# 容器内
cd /workspace/FastWAM
export TASK=autogen_mixed_charger_white
export CKPT=/workspace/FastWAM/runs/autogen_mixed_charger_white/2026-10-23_11-40-22/checkpoints/weights/step_020000.pt
export DATASET_STATS=/workspace/FastWAM/runs/autogen_mixed_charger_white/2026-10-23_11-40-22/dataset_stats.json
export GPU=0 PORT=8765 STEPS=10 TOKEN=fastwam-dev-token
bash scripts/serve_deploy.sh

# 另一个终端
docker exec -it fastwam-serve bash
cd /workspace/FastWAM
curl -s http://127.0.0.1:8765/readyz
python examples/ws_client.py --url ws://127.0.0.1:8765/v1/ws \
  --token fastwam-dev-token --synthetic \
  --prompt "Plug the charger into the leftmost socket of the power outlet." \
  --repeat 3 --verbose
```

更完整的协议、错误码、TLS、限流说明在 `docs/fastwam_infer-serving_ws_api_zh.md`。
如果需要 HTTP/JSON 而不是 WebSocket，可以在现在这个服务前面加一层很薄的 FastAPI 包装；
当前仓库内置的是 WebSocket + msgpack。
`