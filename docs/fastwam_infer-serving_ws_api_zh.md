# FastWAM 云端推理 WebSocket 服务

把 FastWAM 的 action-only 推理（`infer_action`）封装成一个常驻服务：GPU 机器加载
MoT 模型 + 文本编码器，端侧机器人只需要通过 **WebSocket（生产环境 wss）** 发送
「相机图像 + 机械臂状态 + 任务指令」，就能拿回一个 action chunk。

```
              ┌──────────────────────────── 云端 GPU ────────────────────────────┐
 相机 JPEG ──▶│  FastAPI /v1/ws  ──▶  InferenceWorker（单线程，独占 CUDA）       │
 状态向量  ──▶│      │                     │                                  │
 prompt    ──▶│      │                     ├─ processor 归一化 / 拼图            │
              │      │                     ├─ text encoder（prompt LRU 缓存）    │
              │      │                     ├─ model.infer_action(...)            │
              │      ◀─────────────────────┴─ normalizer.backward → action chunk │
              └──────────────────────────────────────────────────────────────────┘
```

- 传输：WebSocket + **二进制 msgpack**，图片直接传 JPEG/PNG 原始字节（不做 base64）。
- 返回：只返回 action chunk，不返回想象视频。
- 配置：由数据集配置驱动（`shape_meta` + `dataset_stats.json`），不写死 LIBERO / RoboTwin。
- 部署：单卡单进程；结构上预留多 worker / 多卡扩展（见第 11 节）。

---

## 1. 快速开始

```bash
# 依赖（一次性）
pip install -e '.[serving]'

# 用真实权重启动
python scripts/serve.py \
  task=mydata_uncond_2cam224_1e-4 \
  serving.ckpt=./runs/mydata_uncond_2cam224_1e-4/<run_id>/checkpoints/weights/20000.pt \
  serving.dataset_stats_path=./runs/mydata_uncond_2cam224_1e-4/<run_id>/dataset_stats.json

# 或者用启动脚本（开发 / 部署两种，区别见 7.1 节）
CKPT=... DATASET_STATS=... bash scripts/serve_dev.sh mydata_uncond_2cam224_1e-4
```

无权重时可用 dry-run 验证传输层（返回确定性的 `sin` 形状 action，不加载模型、不占显存）：

```bash
python scripts/serve.py task=libero_uncond_2cam224_1e-4 serving.dry_run=true
python examples/ws_client.py --synthetic --prompt "put the bowl on the plate" --repeat 5 --verbose
```

健康检查：

```bash
curl http://127.0.0.1:8765/healthz   # {"status":"ok"}           进程存活
curl http://127.0.0.1:8765/readyz    # {"ready":true}            模型加载 + warmup 完成
curl http://127.0.0.1:8765/v1/info   # 模型元信息 + 服务端限制
```

---

## 2. 前置条件：数据配置

服务端从 hydra 配置推导**全部**输入输出维度，端侧无需硬编码。`task=<name>` 选中的
`configs/task/<name>.yaml` 必须提供：

| 字段 | 位置 | 用途 |
|---|---|---|
| `data.train.shape_meta.images[*].key/shape` | 数据配置 | 相机 key 与 `[3,H,W]`；顺序即请求中图片的推荐顺序 |
| `data.train.shape_meta.state[*]` | 数据配置 | state key 与原始维度 `raw_shape`（拼成 `state_dim`） |
| `data.train.shape_meta.action[*]` | 数据配置 | action key 与维度（拼成 `action_dim`） |
| `data.train.video_size` | 数据配置 | 拼图后的 `[H,W]`，必须是 16 的倍数 |
| `data.train.concat_multi_camera` | 数据配置 | `horizontal` / `vertical` / `robotwin` |
| `data.train.num_frames` | 数据配置 | `action_horizon = num_frames - 1`（可被 `serving.action_horizon` 覆盖） |
| `data.train.processor` | 数据配置 | 归一化/反归一化逻辑，服务端复用训练侧 processor |
| `dataset_stats.json` | 命令行 | 与 checkpoint 对应的归一化统计量 |

服务启动时会校验「各相机 shape 拼接后 == `video_size`」，不匹配直接报错（避免带病上线）。
实际生效的值通过 `hello_ack.model` / `/v1/info` 下发。

> 自定义数据集只要满足上述 `shape_meta` + `dataset_stats.json` 约定即可直接使用；
> 拼图方式目前支持 `horizontal`、`vertical`、`robotwin` 三种。

---

## 3. 协议规范 `fastwam.ws.v1`

### 3.1 连接与帧格式

```
ws://{host}:{port}/v1/ws          （开发 / 内网）
wss://{host}:{port}/v1/ws?token=… （生产，见第 8 节）
```

- 每个 **二进制** 帧 = 一个 msgpack map：`msgpack.packb(obj, use_bin_type=True)`。
- 所有帧带 `v: 1` 和 `type`；首个客户端帧必须是 `hello`。
- 不支持文本帧（返回 `UNSUPPORTED_FORMAT`）。
- 单帧上限 `serving.max_message_bytes`（默认 16 MiB），超限返回错误并以 close code `1009` 关闭。
- 只使用 msgpack 的 map / array / str / int / float / bool / bin 六种类型，浮点为 double，
  任意语言的 msgpack 库都能解析。

服务端在每个连接内**串行**处理请求（发一个 `infer`、收一个 `actions`），不同连接由
推理队列并发调度。请用 `id` 字段做请求/响应配对。

### 3.2 鉴权

- 设置环境变量 `FASTWAM_SERVICE_TOKEN` 即启用（`serving.auth_token_env` 可改名）。
- 客户端通过 `?token=` 查询参数或 `hello.token` 提供，服务端用 `hmac.compare_digest` 比较。
- 未设置该变量时服务照常启动，但会打印警告并允许匿名访问 —— **仅限可信内网 / 开发环境**。
- 鉴权失败：返回 `UNAUTHORIZED` 错误帧，并以 close code `4401` 关闭。

### 3.3 握手：`hello` → `hello_ack`

客户端 → 服务端：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `v` | int | 是 | 固定 `1` |
| `type` | str | 是 | `"hello"` |
| `token` | str | 否 | 未放在 query 时使用 |
| `client` | str | 否 | 客户端标识，仅用于日志 |
| `prompt` | str | 否 | 会话默认任务指令 |
| `options` | map | 否 | 会话默认 options，字段同 3.4 |

服务端 → 客户端：

```python
{
  "v": 1, "type": "hello_ack",
  "server": "fastwam/0.1.0",
  "session_id": "8f2c1e...",
  "model": {
      "variant": "FastWAM",            # dry-run 时为 "dry-run"
      "action_dim": 7,
      "state_dim": 8,
      "action_horizon": 32,
      "video_size": [224, 448],        # [H, W]
      "concat_multi_camera": "horizontal",
      "cameras": [{"key": "image",       "shape": [3, 224, 224]},
                  {"key": "wrist_image", "shape": [3, 224, 224]}],
      "action_keys": ["default"],      # 输出向量按此顺序拼接
      "state_keys":  ["default"],      # 输入 state 按此顺序拼接
      "normalization": "min/max",
      "uses_proprio": True,
      "dry_run": False,
      "compile_action_infer": True
  },
  "limits": {
      "max_images": 8,
      "max_image_bytes": 4194304,
      "max_state_dim": 8,
      "queue_depth": 16,
      "request_timeout_ms": 5000
  }
}
```

### 3.4 推理请求：`infer`

```python
{
  "v": 1, "type": "infer",
  "id": "req-000123",                    # str 或 int，原样回传
  "prompt": "put the bowl on the plate", # 可选；hello 已给默认 prompt 时可省略
  "images": [
      {"key": "image",       "format": "jpeg", "data": b"\xff\xd8..."},
      {"key": "wrist_image", "format": "jpeg", "data": b"\xff\xd8..."}
  ],
  "state": [0.12, -0.03, ...],           # 原始（未归一化）state
  "options": {
      "num_inference_steps": 10,
      "sigma_shift": None,               # None → scheduler 默认；旧版 release 权重常用 5.0
      "seed": None,                      # 固定后动作可复现
      "action_horizon": 32,
      "timeout_ms": 5000
  }
}
```

约束：

- `images` 数量必须等于 `hello_ack.model.cameras` 的个数。
- `images[i].key` 可省略：全部省略时按**位置**对应；给了 key 时按 key 匹配（与顺序无关），
  缺 key 或 key 不匹配 → `BAD_REQUEST`。
- `format` 只接受 `jpeg` / `jpg` / `png`，服务端用 PIL 解码并校验真实格式；失败 → `BAD_REQUEST`。
- `state` 长度必须等于 `state_dim`，按 `state_keys` 顺序拼接（服务端始终校验长度）。
  不使用 proprio 的模型在 `hello_ack.model.uses_proprio=false` 时忽略该字段内容，
  但仍需发送占位的 `state_dim` 个值。
- `prompt` 与 hello 的默认 prompt 二选一，都缺失 → `BAD_REQUEST`。
- `options` 中为 `None` 的字段回退到 hello 的会话默认值，再回退到服务端配置。
- `timeout_ms` 是「排队 + 推理」的总预算，超时返回 `TIMEOUT`（后台线程无法强杀，
  该次计算仍会跑完，结果丢弃）。

### 3.5 响应：`actions`

```python
{
  "v": 1, "type": "actions",
  "id": "req-000123",
  "actions": [[0.01, ...], ...],   # [action_horizon][action_dim]，float
  "action_horizon": 32,
  "action_dim": 7,
  "action_keys": ["default"],
  "space": {"state": "dataset observation.state", "action": "dataset action"},
  "seed_used": 42,                 # 实际使用的 seed（未指定则为 null）
  "timing": {
      "queue_ms": 0.2,             # 在推理队列中等待的时间
      "preprocess_ms": 2.1,        # 解码 / 裁剪 / 拼图 / 归一化
      "text_ms": 0.0,              # prompt 编码（命中 LRU 缓存时为 0）
      "infer_ms": 88.5,            # VAE encode + 动作去噪（含 CUDA 同步）
      "postprocess_ms": 0.6,       # 反归一化
      "total_ms": 101.7
  },
  "server_ts": 1760000000.123
}
```

### 3.6 心跳与重置

- `{"v":1,"type":"ping","ts":...}` → `{"v":1,"type":"pong","ts":...}`
- `{"v":1,"type":"reset"}` → 清空该连接的默认 prompt / options，回 `{"type":"reset_ack","session_id":...}`

### 3.7 错误

```python
{"v": 1, "type": "error", "id": "req-000123",
 "code": "BAD_REQUEST", "message": "...", "detail": "..."}
```

| code | 触发场景 | 连接是否保持 |
|---|---|---|
| `BAD_REQUEST` | msgpack 解码失败、字段缺失/类型错误、图片数量/格式/state 维度不符、缺少 prompt | 保持 |
| `UNSUPPORTED_FORMAT` | 文本帧、未知 `type`、首帧不是 `hello`、帧过大 | 首帧错误 / 帧过大时关闭 |
| `UNAUTHORIZED` | token 缺失或错误 | 关闭（4401） |
| `BUSY` | 队列已满（`limits.queue_depth`） | 保持 |
| `TIMEOUT` | 排队 + 推理超过 `timeout_ms` | 保持 |
| `INTERNAL` | 推理异常（服务端记录 traceback） | 保持 |

连接建立但模型未就绪时，握手返回 `INTERNAL: model is not ready` 并关闭（1013）。

---

## 4. 图像预处理与相机顺序

服务端按 `concat_multi_camera` 把多路相机拼成一张 `video_size` 的图像：

| 模式 | 处理方式 |
|---|---|
| `horizontal` | 每路 `center_crop_resize` 到 `shape_meta` 的 `[3,H,W]`，再按宽度横向拼接 |
| `vertical` | 同上，按高度纵向拼接 |
| `robotwin` | 3 路相机：head → `320×256`，左右腕 → `160×128`，下排拼成 `320×128`，上下一拼得到 `384×320`（与 RoboTwin 评测脚本一致） |

`center_crop_resize` 保持长宽比缩放后中心裁剪，与 `examples/infer_custom.py` 的 LIBERO
评测逻辑完全相同 —— 端侧只需发送原始相机图（任意分辨率），**不需要**自己 resize。

> 注意：RoboTwin 训练侧是先 resize 到 `shape_meta` 再 mosaic，推理侧沿用评测脚本的直接
> resize；这与已发布结果的评测口径一致。

---

## 5. 动作空间契约（重要）

返回的 `actions` 是**数据集 action 空间**、按 `shape_meta.action` 顺序拼接的向量，
服务端只做 `normalizer.backward`（反归一化）：

- **不做** delta ↔ absolute 转换；
- **默认不**逆 `action_state_transform`（与现有 eval 脚本一致，`serving.invert_action_transforms` 预留）。

因此端侧必须按训练数据的语义执行，例如：

- LIBERO：delta EEF 位移 + 绝对 gripper；
- RoboTwin：绝对关节 qpos。

服务端同样**不**做平滑、限幅、插值或时间对齐，这些属于端侧策略层。

---

## 6. 配置项（`configs/serve.yaml` → `serving.*`）

| 配置 | 默认 | 说明 |
|---|---|---|
| `device` | `cuda` | `cuda` / `cuda:N` / `cpu` |
| `dtype` | `bf16` | `bf16` / `fp16` / `fp32` |
| `ckpt` | `null` | checkpoint 路径；`dry_run=false` 时必填 |
| `dataset_stats_path` | `null` | `dataset_stats.json`；`dry_run=false` 时必填 |
| `host` / `port` | `0.0.0.0` / `8765` | 监听地址 |
| `compile_action_infer` | `true` | `torch.compile(mode="reduce-overhead", fullgraph=True)` 加速动作分支 |
| `force_serve_overrides` | `true` | 强制 `load_text_encoder=true`、`skip_dit_load_from_pretrain=true`、`action_dit_pretrained_path=null` |
| `text_encoder_device` | `cuda` | 改 `cpu` 可省约 10 GB 显存，代价是每次新 prompt 慢一些 |
| `num_inference_steps` | `10` | 默认去噪步数（可被请求覆盖） |
| `sigma_shift` | `null` | `null` 用 scheduler 默认；旧版 release 权重常用 `5.0` |
| `action_horizon` | `null` | `null` → `data.train.num_frames - 1` |
| `num_video_frames` | `null` | joint/idm 变体使用；`null` → 按 `action_video_freq_ratio` 推导 |
| `action_infer_mode` | `first_frame` | `first_frame` / `idm`（仅 Optional-IDM 变体） |
| `invert_action_transforms` | `false` | 预留，默认与 eval 脚本一致 |
| `prompt_cache_size` | `64` | prompt 文本 embedding 的 LRU 上限 |
| `max_queue` | `16` | 推理队列长度；满了返回 `BUSY` |
| `request_timeout_ms` | `5000` | 默认「排队 + 推理」超时 |
| `init_timeout_s` | `1800` | 加载 + `torch.compile` + warmup 的启动超时 |
| `max_message_bytes` | `16777216` | 单帧上限（同时用于 uvicorn `ws_max_size`） |
| `max_images` | `8` | 单请求最多图片数 |
| `max_image_bytes` | `4194304` | 单张图片字节上限 |
| `warmup` / `warmup_runs` | `true` / `2` | 启动时预热，消化编译与 CUDA graph capture |
| `dry_run` | `false` | 不加载权重，返回合成动作，用于传输层联调 |
| `auth_token_env` | `FASTWAM_SERVICE_TOKEN` | 读取 token 的环境变量名 |
| `log_level` | `INFO` | 日志级别 |
| `tls_certfile` / `tls_keyfile` | `null` | 进程内 TLS；推荐用反向代理终止 TLS |

---

## 7. 部署

### 7.1 启动脚本怎么选：`serve_dev.sh` vs `serve_deploy.sh`

两个脚本最终都执行同一个 `scripts/serve.py`、传同一组 hydra override，区别在定位：

| 维度 | `scripts/serve_dev.sh` | `scripts/serve_deploy.sh` |
|---|---|---|
| 定位 | 开发 / 单次调试 | 部署 / 运维 |
| 参数风格 | 第一个位置参数是 `<task>`，其余透传 hydra | 全部用环境变量，`TASK` 有默认值 |
| 必填项 | `CKPT`、`DATASET_STATS`（`DRY_RUN=1` 时免） | 同左 |
| 文件校验 | 只检查是否为空 | 还检查 `configs/task/<TASK>.yaml`、`CKPT`、`DATASET_STATS` 是否真实存在 |
| GPU 选择 | 不支持（自行 export） | `GPU=0` → 自动 `CUDA_VISIBLE_DEVICES` |
| 推理参数 | 只能透传 hydra 覆盖 | `STEPS`、`SIGMA_SHIFT`、`ACTION_HORIZON`、`COMPILE`、`DTYPE`、`TEXT_ENCODER_DEVICE`、`MAX_QUEUE`、`REQUEST_TIMEOUT_MS` 直接变量化 |
| 鉴权 | 只读 `FASTWAM_SERVICE_TOKEN` | 支持 `TOKEN` 或 `TOKEN_FILE`（默认 `/etc/fastwam/token`），自动 export |
| 运行方式 | 仅前台 `exec` | 前台，或 `BACKGROUND=1` 用 nohup 后台运行 |
| 日志 / PID | 无（交给调用方重定向） | 自动写 `logs/serve_<TASK>_<时间>.log` 和 `.pid` |
| 就绪等待 | 无 | 后台模式自动轮询 `/readyz`（`WAIT_TIMEOUT`），进程提前退出会打印日志尾部并返回非 0 |
| 启动提示 | 打印参数摘要 | 额外打印健康检查和 `ws_client.py` 测试命令 |
| 代码量 | ~40 行 | ~200 行 |

选择建议：

- 日常开发 / 调参：用 `serve_dev.sh`，参数少、直接透传 hydra；
- 部署 / tmux / systemd / 容器：用 `serve_deploy.sh`，有文件校验、后台日志、PID 与就绪门控。

### 7.2 直接启动 + `serve_deploy.sh`

```bash
FASTWAM_SERVICE_TOKEN=$(cat /etc/fastwam/token) \
python scripts/serve.py task=... serving.ckpt=... serving.dataset_stats_path=...
```

或使用变量驱动脚本 `scripts/serve_deploy.sh`（会校验权重/统计文件是否存在、设置 GPU、
可选 token 文件与后台运行）：

```bash
# 前台：Ctrl-C 停止
CKPT=/data/fastwam/weights/20000.pt \
DATASET_STATS=/data/fastwam/dataset_stats.json \
TASK=mydata_uncond_2cam224_1e-4 \
GPU=0 PORT=8765 \
  bash scripts/serve_deploy.sh

# 后台常驻：日志与 pid 写到 logs/，脚本会等 /readyz 后再返回
CKPT=... DATASET_STATS=... TASK=... BACKGROUND=1 \
TOKEN_FILE=/etc/fastwam/token \
  bash scripts/serve_deploy.sh

# 无权重联调
DRY_RUN=1 PORT=8765 bash scripts/serve_deploy.sh
```

开发调试也可以直接用 `serve_dev.sh`（前台、`<task>` 位置参数、剩余参数透传 hydra）：

```bash
CKPT=... DATASET_STATS=... bash scripts/serve_dev.sh mydata_uncond_2cam224_1e-4
DRY_RUN=1 bash scripts/serve_dev.sh libero_uncond_2cam224_1e-4 serving.num_inference_steps=6
```

可用变量（也可写在脚本头部）：`TASK`、`CKPT`、`DATASET_STATS`、`GPU`、`HOST`、`PORT`、
`DTYPE`、`TEXT_ENCODER_DEVICE`、`STEPS`、`SIGMA_SHIFT`、`ACTION_HORIZON`、`COMPILE`、
`MAX_QUEUE`、`REQUEST_TIMEOUT_MS`、`TOKEN`/`TOKEN_FILE`、`DRY_RUN`、`BACKGROUND`、
`LOG_FILE`、`WAIT_READY`/`WAIT_TIMEOUT`、`PYTHON_BIN`、`EXTRA_ARGS`；命令行剩余参数
（如 `serving.sigma_shift=5.0`）会透传给 hydra，且优先级最高。

`/readyz` 返回 200 前请勿把流量切到该实例（加载 + `torch.compile` 可能需要数分钟）。

### 7.3 Docker

镜像已包含 `[serving]` 依赖并 `EXPOSE 8765`（见 `docker/Dockerfile`）：

```bash
docker run --gpus all --rm -p 8765:8765 \
  -e FASTWAM_SERVICE_TOKEN=... \
  -e TASK=mydata_uncond_2cam224_1e-4 \
  -e CKPT=/workspace/checkpoints/weights/20000.pt \
  -e DATASET_STATS=/workspace/checkpoints/dataset_stats.json \
  -e GPU=0 \
  -v /data/fastwam:/workspace/checkpoints \
  fastwam:cu126 \
  bash scripts/serve_deploy.sh
```

> `docker/Dockerfile` 自带 torch **cu126** wheel，只编译到 sm_90。RTX 50 系列（sm_120）
> 会报 `no kernel image is available`，需换成 cu128 wheel（`pyproject.toml` 的默认 pin
> 就是 `torch==2.7.1+cu128` / `torchvision==0.22.1+cu128`）。

### 7.4 wss（TLS）

推荐用反向代理终止 TLS，服务只监听 `127.0.0.1`：

nginx：

```nginx
location /v1/ws {
    proxy_pass http://127.0.0.1:8765;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 3600s;
}
```

Caddy：

```
robot.example.com {
    reverse_proxy 127.0.0.1:8765
}
```

也可以进程内启用：

```bash
python scripts/serve.py ... serving.tls_certfile=/etc/tls/fullchain.pem serving.tls_keyfile=/etc/tls/privkey.pem
```

---

## 8. 客户端

参考实现：`examples/ws_client.py`（异步 `websockets`，支持真实图片 / 合成图片 / 压测 / 输出 JSON）。

```bash
# 真实图像
python examples/ws_client.py \
  --url ws://127.0.0.1:8765/v1/ws --token "$TOKEN" \
  --image cam_high.jpg --image cam_wrist.jpg \
  --state 0.1,0.2,0.3,0.0,0.0,0.0,0.04,0.04 \
  --prompt "put the bowl on the plate" --output /tmp/actions.json

# 传输层联调 / 压测
python examples/ws_client.py --synthetic --prompt "pick up the cube" --repeat 50
```

最小 Python 客户端：

```python
import asyncio, msgpack
from websockets.asyncio.client import connect

def pack(o):   return msgpack.packb(o, use_bin_type=True)
def unpack(b): return msgpack.unpackb(b, raw=False, strict_map_key=False)

async def main():
    async with connect("ws://127.0.0.1:8765/v1/ws?token=TOKEN") as ws:
        await ws.send(pack({"v": 1, "type": "hello", "client": "robot", "prompt": "pick up the cube"}))
        ack = unpack(await ws.recv())
        cameras = ack["model"]["cameras"]

        await ws.send(pack({
            "v": 1, "type": "infer", "id": "t0",
            "images": [{"key": c["key"], "format": "jpeg", "data": jpeg_bytes(c["key"])}
                       for c in cameras],
            "state": state.tolist(),          # state_dim 个原始值
            "options": {"seed": 42},
        }))
        resp = unpack(await ws.recv())
        if resp["type"] == "error":
            raise RuntimeError(f"{resp['code']}: {resp['message']}")
        actions = resp["actions"]             # [horizon][action_dim]
```

---

## 9. 性能与调优

- **warmup**：启动时跑 `warmup_runs` 次；`compile_action_infer=true` 的首次编译与 CUDA graph
  capture 都在 warmup 内完成，`/readyz` 在此之前返回 503。
- **prompt 缓存**：同一任务的文本 embedding 只编码一次（LRU 上限 `prompt_cache_size`）。
  端侧应把任务指令放在 `hello.prompt` 里复用。
- **`num_inference_steps`**：步数直接决定 `infer_ms`；10 步是默认值。
- **显存**：bf16 下 video DiT ~10 GB + T5 ~11 GB + action DiT ~2 GB + VAE（合计约 23–24 GB）。
  24 GB 卡建议 `serving.text_encoder_device=cpu`；固定指令集场景也可预计算 context。
- **并发**：单 worker 串行使用 GPU。客户端并发时延迟约等于排队时间 + 单次推理时间；
  `timing.queue_ms` 上升说明需要更多副本（多卡/多实例）。
- **客户端帧发送**：一帧一张推理请求，不要流水线式连发多帧后不读响应 —— 服务端按连接串行处理。

参考客户端实测（dry-run，宿主 load average >100 的共享机器上）：单连接 p50 ~7 ms；
两个并发连接共 40 次请求无串号、无错误。

---

## 10. 多卡扩展

当前实现是**单卡单进程**：一个 `InferenceWorker` 线程独占一张 GPU。扩展方式：

1. **多进程**：每个 GPU 起一份 `scripts/serve.py`（不同 `serving.port` / `CUDA_VISIBLE_DEVICES`），
   前置 nginx / 网关按「会话粘滞」（同一机器人固定到某实例）分流。
2. **多 worker 线程**：同一进程内每个 GPU 一个 `InferenceWorker`，`app.py` 按会话选择 worker。
   由于每张卡只被一个线程使用，`torch.compile` / CUDA graph 不存在跨线程问题。

单卡内**不要**用 `asyncio.to_thread` 直接并发进 CUDA：多流会争抢显存并可能破坏
CUDA graph 的状态。

---

## 11. 排障

| 现象 | 原因 / 处理 |
|---|---|
| `/readyz` 一直 503 | 看服务日志：权重路径、`dataset_stats.json`、显存不足、`torch.compile` 失败 |
| `CUDA error: no kernel image is available` | torch 版本不含该 GPU 架构的 kernel（典型：cu126 + RTX 5090/sm_120），换 cu128 wheel |
| 握手返回 `INTERNAL: model is not ready` | 初始化失败；`/readyz` 的 `error` 字段有异常摘要 |
| `BAD_REQUEST: expected N images` | 图片数量与 `hello_ack.model.cameras` 不一致 |
| `BAD_REQUEST: state must have N values` | state 维度与 `state_dim` 不一致（注意多 key 拼接顺序） |
| 动作尺度明显不对 | 检查 `dataset_stats_path` 是否与 checkpoint 匹配；检查 `sigma_shift`、`action_horizon` |
| 动作与离线脚本不一致 | 依次核对：相机顺序、`video_size`/拼图模式、state 归一化顺序、prompt 模板、seed |
| 请求偶发 `TIMEOUT` | `queue_ms` 高 → 并发超过单卡吞吐；适当提高 `request_timeout_ms` 或加副本 |
| 帧过大被关闭（1009） | 调大 `serving.max_message_bytes`，或客户端压缩图片 |

### 已知限制

- 服务端**不**校验动作的安全性（限幅、碰撞），端侧必须自行兜底。
- 请求超时后后台计算仍会跑完（线程无法强杀），高负载时请配合 `max_queue` 使用。
- 目前只支持 `horizontal` / `vertical` / `robotwin` 三种拼图；自定义布局需要新增分支。
