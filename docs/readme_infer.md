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
- [5. 请求 body 示例](#5-请求-body-示例)
- [6. 机器人侧客户端](#6-机器人侧客户端)
- [7. 后台常驻与停止](#7-后台常驻与停止)
- [8. 常见问题](#8-常见问题)
- [9. 最短路径总结](#9-最短路径总结)

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
bash ./docker/run_container.sh
```

几点说明：

- `HOST_RUNS_DIR` 必须指向 `docker_dir/runs`，否则容器里看不到 `step_020000.pt`。
- `HOST_DATA_DIR` 用 `train_data` 只是为了下面从训练数据里抽两张真实图做测试；
  **服务本身不读数据集**。
- `GPUS='"device=0"'` 只暴露 0 号卡；如果 0 号卡被占用，改成别的编号。
- 容器名会是 `fastwam-serve`。如果已存在同名容器，先 `docker rm -f fastwam-serve`。
- `--network=host` 已由脚本设置，所以宿主机直接访问 `127.0.0.1:8765`。


### 1.3 容器内补装 serving 依赖（只有旧镜像需要）

按当前 `docker/Dockerfile` 构建的镜像**已经包含** `[serving]` 依赖
（`fastapi` / `uvicorn[standard]` / `msgpack` / `websockets`），这一步可以跳过。

2026-10-15 那版镜像早于 Dockerfile 加入 `[serving]`，容器里没有这些包，需要手工补装：

```bash
docker exec -u 0 <容器名> bash -c 'cat >> /etc/profile.d/proxy.sh <<EOF
export https_proxy=http://127.0.0.1:17897
export http_proxy=http://127.0.0.1:17897
export no_proxy=localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
EOF'

## 宿主机外直接安装（必须装 extra，裸 uvicorn 不带 WebSocket 实现）
docker exec -u 0 <容器名> bash -lc '
export https_proxy=http://127.0.0.1:17897 http_proxy=http://127.0.0.1:17897 all_proxy=http://127.0.0.1:17897
cd /workspace/FastWAM && python -m pip install -e ".[serving]"'
```

注意两点：

- 用 `python -m pip` 而不是裸 `pip`，确保装进 `scripts/serve.py` 实际用的解释器
  （镜像里是 `/opt/venv/bin/python`）。
- **不要用 `pip install uvicorn`**：裸 `uvicorn` 不含 `websockets`，服务仍能起来，
  但 `/v1/ws` 的协议升级请求会被当成普通 `GET` 处理并返回 404，客户端报
  `websockets.exceptions.InvalidStatus: server rejected WebSocket connection: HTTP 404`，
  服务端日志是 `No supported WebSocket library detected.`。手工装进去的包在容器重建后
  也会丢，正式做法是按 `docker/Dockerfile` 重建镜像。

---

## 2. 容器内自检

进入容器后：

```bash
cd /workspace/FastWAM

python -c "import torch, fastwam; print(torch.__version__, torch.cuda.is_available()); print(fastwam.__file__)"

# serving 依赖 + WebSocket 后端（旧镜像这里会 ImportError，见 1.3）
python -c "import fastapi, uvicorn, msgpack, websockets; from uvicorn.protocols.websockets.auto import AutoWebSocketsProtocol as ws; print('serving ok', fastapi.__version__, uvicorn.__version__, websockets.__version__, ws.__name__)"
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

## huashu
export TASK=autogen_mixed_charger_white
export CKPT=/workspace/FastWAM/runs/autogen_mixed_charger_white/2026-10-23_11-40-22/checkpoints/weights/step_020000.pt
export DATASET_STATS=/workspace/FastWAM/runs/autogen_mixed_charger_white/2026-10-23_11-40-22/dataset_stats.json

## ur5数据
export TASK=autogen_mixed_ur5
export CKPT=/workspace/FastWAM/runs/autogen_mixed_ur5/2026-10-27_02-19-37/checkpoints/weights/step_030000.pt
export DATASET_STATS=/workspace/FastWAM/runs/autogen_mixed_ur5/2026-10-27_02-19-37/dataset_stats.json


## guodi
export TASK=autogen_mixed_guodi_wrc
export CKPT=/workspace/FastWAM/runs/autogen_mixed_guodi_wrc/2026-10-30_07-43-00/checkpoints/weights/step_030000.pt
export DATASET_STATS=/workspace/FastWAM/runs/autogen_mixed_guodi_wrc/2026-10-30_07-43-00/dataset_stats.json



## flexi
export TASK=autogen_mixed_lerobot_v30_flexiv_hz_union_v1
export CKPT=/workspace/FastWAM/runs/autogen_mixed_lerobot_v30_flexiv_hz_union_v1/2026-10-28_10-18-11/checkpoints/weights/step_030000.pt
export DATASET_STATS=/workspace/FastWAM/runs/autogen_mixed_lerobot_v30_flexiv_hz_union_v1/2026-10-28_10-18-11/dataset_stats.json


## xingsong
export TASK=autogen_mixed_guodi_wrc
export CKPT=/workspace/FastWAM/runs/autogen_mixed_xingsong_resort_join_gripper/2026-10-26_11-15-32/checkpoints/weights/step_030000.pt
export DATASET_STATS=/workspace/FastWAM/runs/autogen_mixed_xingsong_resort_join_gripper/2026-10-26_11-15-32/dataset_stats.json

############################## 启动指令
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
DS=/mnt/sfs_turbo/h00672373/train_data/hs_data_v30/lerobot_charger_0324
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
    '/mnt/sfs_turbo/h00672373/train_data/hs_data_v30/lerobot_charger_0324/data/chunk-000/file-000.parquet',
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

## 5. 请求 body 示例

服务端没有 HTTP 那种 URL + JSON body：**每一帧就是一个 msgpack map**
（`msgpack.packb(frame, use_bin_type=True)`），走 WebSocket 二进制帧。
下面用 JSON 展示帧的**结构**只是方便阅读，实际发送必须用 msgpack 编码，
并且 `images[i].data` 放的是**图片文件的原始字节**，不是 base64 字符串。

两个最容易踩的约束：

- **不能拼错字段名，也不能多加字段**。请求模型是 pydantic `extra="forbid"`，
  多一个未知键（比如 `image` 写成 `images`、`num_steps` 写成 `num_inference_steps`）
  会直接返回 `BAD_REQUEST: invalid infer frame`。
- `hello` 是**可选**的：不握手时把 token 放在 URL（`?token=...`），并且每次 `infer`
  都要带 `prompt`（没有会话默认值可用）。

### 5.1 `hello`（可选）：设置会话默认值

```json
{
  "v": 1,
  "type": "hello",
  "token": "fastwam-dev-token",
  "client": "hsr602",
  "prompt": "Plug the charger into the leftmost socket of the power outlet.",
  "options": {
    "num_inference_steps": 10
  }
}
```

| 字段 | 必填 | 说明 | 本任务取值 |
|---|---|---|---|
| `v` | 是 | 协议版本，固定 `1` | `1` |
| `type` | 是 | 固定 `"hello"` | `"hello"` |
| `token` | 否 | 服务端设了 `TOKEN` 才需要；也可以放在 URL `?token=...` | `fastwam-dev-token` |
| `client` | 否 | 客户端标识，只进日志 | 任意 |
| `prompt` | 否 | 会话默认指令；`infer` 里带 `prompt` 时会覆盖它，所以只跑固定任务时才需要填 | 见下方两条指令 |
| `options` | 否 | 会话默认推理参数，字段同 5.2 的 `options` | `{"num_inference_steps": 10}` |

握手的作用是省掉每次请求里的 `prompt` / `options`，并拿回 `hello_ack.model`
（相机 key/顺序/尺寸、`state_dim`、`action_dim`、`action_horizon`）和 `limits`。
不需要这些信息时，可以整帧省略，见 5.4。

`prompt` 必须是当前场景对应的指令，这个 checkpoint 只认这两条：

```text
Plug the charger into the leftmost socket of the power outlet.
wipe the charactor 'T' on the whiteboard at the end of 'CloudRoboT'.
```

### 5.2 `infer`：每次推理一帧

```json
{
  "v": 1,
  "type": "infer",
  "id": "req-000123",
  "prompt": "Plug the charger into the leftmost socket of the power outlet.",
  "images": [
    {"key": "front", "format": "jpeg", "data": "<front.jpg 的原始字节>"},
    {"key": "wrist", "format": "jpeg", "data": "<wrist.jpg 的原始字节>"}
  ],
  "state": [
    0.4123, -0.0021, 0.3012,
    0.0110, -0.0004, 0.0521,
    0.0620,
    0.0000, 0.0000, -1.2345,
    0.0021, -0.0451, 0.0000
  ],
  "options": {
    "num_inference_steps": 10
  }
}
```

（`state` 里的数值只是占位示例，别人的机器人要填自己的真实观测。）

| 字段 | 必填 | 说明 |
|---|---|---|
| `v` | 是 | 固定 `1` |
| `type` | 是 | 固定 `"infer"` |
| `id` | 是 | `str` 或 `int`，服务端在 `actions` / `error` 里原样回传，用来对请求 |
| `prompt` | 否 | 本次请求的任务指令；**每次 `infer` 都可以传**，覆盖 `hello.prompt`；不带则用会话默认值，两处都缺 → `BAD_REQUEST` |
| `images` | 是 | 数组，长度必须等于 `hello_ack.model.cameras` 的个数（本任务 2） |
| `images[i].key` | 否 | `"front"` / `"wrist"`；全部省略则按数组**位置**对应，全部给出则按 key 匹配；只给一部分 → `BAD_REQUEST` |
| `images[i].format` | 否 | `jpeg` / `jpg` / `png`，默认 `jpeg`；服务端用 PIL 解码并校验真实格式 |
| `images[i].data` | 是 | 图片原始字节（msgpack bin 类型），单张上限 `limits.max_image_bytes`（默认 4 MB） |
| `state` | 是 | 13 个**未归一化**的原始值，长度必须等于 13 |
| `options` | 否 | 见下表；为 `None` 的字段依次回退到 hello options、服务端配置 |

`options` 可调项：

| 字段 | 范围 | 说明 |
|---|---|---|
| `num_inference_steps` | 1–200 | 去噪步数，默认 10（和服务端 `STEPS` 一致） |
| `sigma_shift` | (0, 100] | 默认 `null`，用 scheduler 默认值 |
| `seed` | ≥ 0 | 固定后动作可复现 |
| `action_horizon` | 1–1024 | 默认 32，和训练一致，一般不改 |
| `timeout_ms` | 1–600000 | 单次「排队 + 推理」的总预算，默认取服务端 `REQUEST_TIMEOUT_MS` |

补充约束：

- 整帧上限 `serving.max_message_bytes`（默认 16 MB），超了连接会被关闭（1009）。
- 图片**不用自己 resize / 裁剪 / 横拼**：服务端按 `cameras[].shape` 处理，再拼成 `224x448`。
- 不写 key 时，图片顺序必须是 **front、wrist**，传反了动作会明显不对。
- `state` 13 维和输出动作 13 维的顺序见第 6 节「动作语义」。
- 任务指令在每次 `infer` 里传 `prompt` 即可，同一条连接上切换任务**不需要重连**。
  相同 prompt 会命中服务端 LRU 缓存（`timing.text_ms ≈ 0`）；新 prompt 第一次会
  多一次文本编码开销。

### 5.3 图像尺寸要求（image_size）

- **客户端传原图即可，不用自己 resize / 裁剪**：服务端对每路图做保比例缩放 + 中心裁剪
  到该相机的目标尺寸，再按 `concat_multi_camera` 拼接。
- 目标尺寸来自 task 对应的 data 配置，不是服务端写死的默认值。本任务
  `configs/data/autogen_mixed_charger_white.yaml`：每路 `shape: [3, 224, 224]`，
  两路横向拼成 `video_size: [224, 448]`。换 task 会自动变，可查 `/v1/info` 或
  `hello_ack.model.cameras[].shape`。
- 数据集原始帧是 `raw_shape: [3, 360, 640]`，但**客户端不必按这个尺寸传**，任意尺寸都行。
- 服务端是**中心裁剪**：目标区域尽量居中；原图宽高比和目标（1:1）差太多时边缘会被裁掉。
- 只支持 `jpeg` / `png`，单张 ≤ 4 MB（`serving.max_image_bytes`）。
- 分辨率是训练时定死的，**不能只改推理配置**（改 data 配置里的 `shape` 需重新训练）。

### 5.4 不握手的最小调用

跳过 `hello` 时，第一条消息直接发 `infer` 即可，需要自己保证三件事：

1. **鉴权放 URL**：`ws://<host>:8765/v1/ws?token=fastwam-dev-token`
   （服务端没设 `TOKEN` 时可省略）。
2. **每次 `infer` 都带 `prompt`**：没有会话默认值可用，缺了返回 `BAD_REQUEST`。
3. **自己知道相机和维度**：本任务是 `front`、`wrist`（拍这个顺序）、13 维 state、
   32 步 horizon；不确定就先 `curl /v1/info` 查一次。

```python
frame = {
    "v": 1, "type": "infer", "id": "req-0",
    "prompt": "Plug the charger into the leftmost socket of the power outlet.",
    "images": [{"format": "jpeg", "data": front_bytes},
               {"format": "jpeg", "data": wrist_bytes}],  # 省略 key 时按位置：front、wrist
    "state": state_13,                                    # 13 个原始值
    "options": {"num_inference_steps": 10},
}
await ws.send(msgpack.packb(frame, use_bin_type=True))
```

响应只有 5.7 的 `actions` 帧（不再有 `hello_ack`）。token 错误或缺失时，第一条
`infer` 返回 `UNAUTHORIZED` 并以 4401 断开。`hello` 之后仍然可以随时补发，
重发会覆盖会话默认值。

### 5.5 按需求改哪个字段

| 想做的事 | 改哪里 |
|---|---|
| 换场景 / 任务指令 | 每次 `infer.prompt`（最灵活）；固定任务也可以只在 `hello.prompt` 设一次 |
| 换一帧观测图 | 替换 `images[0].data`（front）、`images[1].data`（wrist） |
| 换成 PNG | `images[i].format = "png"`，`data` 放 PNG 字节 |
| 换机器人状态 | 整体替换 `state`，必须 13 个原始值 |
| 不写 key，按位置传 | 删掉**所有** `images[i].key`（只删一部分会报错） |
| 调推理速度 / 质量 | `options.num_inference_steps`（小更快、大更慢） |
| 让动作可复现 | `options.seed = 1234` |
| 放宽单次超时 | `options.timeout_ms = 20000` |
| 同一条连接切到另一个任务 | 下一次 `infer` 换成新的 `prompt`，不用重连 |

不要做的事：

- 不要把图片转成 base64 / data URL：`data` 必须是原始字节。
- 不要传归一化后的 state：要原始未归一化的值。
- 不要加协议里没有的字段：会直接 `BAD_REQUEST`。

### 5.6 用现成客户端改 body

`examples/ws_client.py` 的 CLI 参数就是对 body 字段的映射：

| CLI 参数 | 对应 body 字段 |
|---|---|
| `--token` | `hello.token`；加 `--no-handshake` 时自动拼到 URL |
| `--prompt` | `hello.prompt` 和每次 `infer.prompt` |
| `--image`（可重复） | `infer.images[i].data`，按出现顺序 |
| `--synthetic` | 用纯色图替换 `images[*].data` |
| `--state` | `infer.state` |
| `--steps` | `options.num_inference_steps` |
| `--seed` | `options.seed` |
| `--sigma-shift` | `options.sigma_shift` |
| `--repeat` | 同一连接上连发 N 次 `infer` |
| `--id-prefix` | `infer.id`（`<prefix>-<index>`） |
| `--no-handshake` | 完全不发 `hello`；此时 `--state` 必须显式给 |
| `--num-cameras` | 无握手 + `--synthetic` 时生成几张图，默认 2 |

不握手跑一次：

```bash
python examples/ws_client.py \
  --url ws://127.0.0.1:8765/v1/ws \
  --token fastwam-dev-token --no-handshake --synthetic \
  --state 0,0,0,0,0,0,0,0,0,0,0,0,0 \
  --prompt "Plug the charger into the leftmost socket of the power outlet." \
  --repeat 3 --verbose
```

想完全控制 body 就照 5.2 自己拼 dict，然后：

```python
import msgpack

frame = {...}  # 5.2 里的结构，data 放读出来的图片字节
await ws.send(msgpack.packb(frame, use_bin_type=True))
```

### 5.7 返回什么

```json
{
  "v": 1,
  "type": "actions",
  "id": "req-000123",
  "actions": [[0.01, "…共 13 个 float…"], "…共 32 行…"],
  "action_horizon": 32,
  "action_dim": 13,
  "action_keys": ["default"],
  "space": {"state": "dataset observation.state", "action": "dataset action"},
  "seed_used": null,
  "timing": {
    "queue_ms": 0.2,
    "preprocess_ms": 2.1,
    "text_ms": 0.0,
    "infer_ms": 88.5,
    "postprocess_ms": 0.6,
    "total_ms": 101.7
  },
  "server_ts": 1760000000.123
}
```

失败时返回：

```json
{"v": 1, "type": "error", "id": "req-000123",
 "code": "BAD_REQUEST", "message": "invalid infer frame", "detail": "..."}
```

除了 `UNSUPPORTED_FORMAT`（未知 `type` / 文本帧 / 帧过大）和 `UNAUTHORIZED`，其他错误都不会断开连接，
改好 body 继续发即可。完整字段和错误码见 `docs/fastwam_infer-serving_ws_api_zh.md` 第 3 节。

---

## 6. 机器人侧客户端

### 6.1 最小实现（不握手）

相机顺序和维度由客户端自己保证：本任务是 `front`、`wrist`、13 维 state、32 步 horizon。

```python
import io

import msgpack
import numpy as np
from PIL import Image
from websockets.asyncio.client import connect

URL = "ws://127.0.0.1:8765/v1/ws?token=fastwam-dev-token"   # 鉴权放 URL
PROMPT = "Plug the charger into the leftmost socket of the power outlet."
STEPS = 10
REPLAN_STEPS = 5


def jpeg(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


async def run(robot):
    async with connect(URL, max_size=64 * 1024 * 1024, ping_interval=None) as ws:
        request_id = 0
        while True:
            obs = robot.get_observation()
            await ws.send(msgpack.packb({
                "v": 1, "type": "infer", "id": f"req-{request_id}",
                "prompt": PROMPT,                                # 每次都要带
                "images": [{"format": "jpeg", "data": jpeg(obs["front"])},
                           {"format": "jpeg", "data": jpeg(obs["wrist"])}],
                "state": [float(x) for x in obs["state"]],
                "options": {"num_inference_steps": STEPS},
            }, use_bin_type=True))

            resp = msgpack.unpackb(await ws.recv(), raw=False, strict_map_key=False)
            if resp["type"] == "error":
                raise RuntimeError(f"{resp['code']}: {resp['message']}")
            actions = np.asarray(resp["actions"], dtype=np.float32)  # [32, 13]
            assert np.isfinite(actions).all()

            for action in actions[:REPLAN_STEPS]:
                robot.step(action)
            request_id += 1
```

### 6.2 完整实现（带握手）

`examples/ws_client.py` 是一次性请求的完整实现（body 字段含义见第 5 节）。机器人闭环可以按下面这个模板接
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
        # 1) 握手：prompt 放 hello 里当默认值（也可以省略，改由下面每次 infer 传）
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
                "prompt": PROMPT,        # 每次都能传；任务变了就改这里，不用重连
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

## 7. 后台常驻与停止

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

## 8. 常见问题

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
| 客户端 `InvalidStatus: HTTP 404`，服务端 `No supported WebSocket library detected` | 镜像没装 serving 依赖（`pip install uvicorn` 不够）。容器内 `python -m pip install -e ".[serving]"` 后重启服务，或按 `docker/Dockerfile` 重建镜像 |

---

## 9. 最短路径总结

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
bash ./docker/run_container.sh

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