"""Reference WebSocket client for the FastWAM inference service.

Examples:

    # one request with real camera images
    python examples/ws_client.py \
        --url ws://127.0.0.1:8765/v1/ws \
        --image cam_high.jpg --image cam_wrist.jpg \
        --state 0.1,0.2,0.3,0.0,0.0,0.0,0.04,0.04 \
        --prompt "put the bowl on the plate"

    # end-to-end transport check without any files (server may run dry_run)
    python examples/ws_client.py --url ws://127.0.0.1:8765/v1/ws \
        --synthetic --prompt "put the bowl on the plate" --repeat 20

    # benchmark: 20 sequential requests, prints p50/p95 latency
    python examples/ws_client.py --synthetic --prompt "pick up the cube" --repeat 20

    # skip the hello handshake: token goes in the URL, prompt in every infer
    python examples/ws_client.py --no-handshake --token t0k3n \
        --image cam_high.jpg --image cam_wrist.jpg \
        --state 0.1,0.2,0.3,0.0,0.0,0.0,0.04,0.04 \
        --prompt "put the bowl on the plate"
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import statistics
import time
from pathlib import Path
from urllib.parse import quote

import msgpack
import numpy as np
from PIL import Image
from websockets.asyncio.client import connect


def pack(payload: dict) -> bytes:
    return msgpack.packb(payload, use_bin_type=True)


def unpack(data: bytes) -> dict:
    return msgpack.unpackb(data, raw=False, strict_map_key=False)


def parse_state(value: str | None) -> list[float]:
    if not value:
        return []
    return [float(part) for part in value.split(",") if part.strip()]


def load_image(path: str) -> dict:
    data = Path(path).read_bytes()
    suffix = Path(path).suffix.lower()
    fmt = "png" if suffix == ".png" else "jpeg"
    return {"format": fmt, "data": data}


def synthetic_images(cameras: list[dict]) -> list[dict]:
    images = []
    for index, camera in enumerate(cameras):
        _, height, width = camera["shape"]
        array = np.zeros((height, width, 3), dtype=np.uint8)
        array[..., index % 3] = 200
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG", quality=90)
        payload = {"format": "jpeg", "data": buffer.getvalue()}
        if camera.get("key") is not None:
            payload["key"] = camera["key"]
        images.append(payload)
    return images


async def request_once(ws, request_id, images, state, prompt, options) -> dict:
    frame = {
        "v": 1,
        "type": "infer",
        "id": request_id,
        "prompt": prompt,
        "images": images,
        "state": state,
        "options": options,
    }
    await ws.send(pack(frame))
    while True:
        response = unpack(await ws.recv())
        if response.get("type") == "actions":
            return response
        if response.get("type") == "error":
            raise RuntimeError(f"{response.get('code')}: {response.get('message')} "
                               f"{response.get('detail') or ''}")
        # pong / reset_ack: ignore


async def run(args) -> None:
    images = [load_image(path) for path in args.image]
    state = parse_state(args.state)

    # Without a hello frame there is no place to put the token, so send it in
    # the query string (the server accepts either).
    url = args.url
    if args.no_handshake and args.token:
        url += ("&" if "?" in url else "?") + f"token={quote(args.token)}"

    async with connect(url, max_size=args.max_size, ping_interval=None) as ws:
        ack = None
        cameras = None
        if not args.no_handshake:
            hello = {"v": 1, "type": "hello", "client": "ws_client"}
            if args.token:
                hello["token"] = args.token
            if args.prompt:
                hello["prompt"] = args.prompt
            await ws.send(pack(hello))

            ack = unpack(await ws.recv())
            if ack.get("type") != "hello_ack":
                raise RuntimeError(f"handshake failed: {ack}")
            if args.verbose:
                print(json.dumps(ack, indent=2, ensure_ascii=False))

            cameras = ack["model"]["cameras"]

        if args.synthetic or not images:
            if cameras is not None:
                images = synthetic_images(cameras)
            else:
                images = synthetic_images(
                    [{"key": None, "shape": [3, 224, 224]}] * args.num_cameras
                )
        if cameras is not None and len(images) != len(cameras):
            raise SystemExit(f"expected {len(cameras)} images, got {len(images)}")
        if not state:
            if ack is None:
                raise SystemExit("--no-handshake requires --state")
            state = [0.0] * int(ack["model"]["state_dim"])

        options = {"num_inference_steps": args.steps}
        if args.seed is not None:
            options["seed"] = args.seed
        if args.sigma_shift is not None:
            options["sigma_shift"] = args.sigma_shift

        latencies = []
        last = None
        for index in range(args.repeat):
            started = time.perf_counter()
            last = await request_once(
                ws, f"{args.id_prefix}-{index}", images, state, args.prompt, options
            )
            latencies.append((time.perf_counter() - started) * 1000.0)
            if args.verbose or index == 0:
                print(
                    f"[{index}] actions={np.asarray(last['actions']).shape} "
                    f"timing={last['timing']}"
                )

        actions = np.asarray(last["actions"], dtype=np.float32)
        action_dim = ack["model"]["action_dim"] if ack else last["action_dim"]
        assert actions.ndim == 2 and actions.shape[1] == action_dim
        assert np.isfinite(actions).all()

        if args.output:
            Path(args.output).write_text(
                json.dumps(
                    {"actions": actions.tolist(), "timing": last["timing"], "ack": ack},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            print(f"Wrote {args.output}")

        if len(latencies) > 1:
            ordered = sorted(latencies)
            p50 = statistics.median(ordered)
            p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
            print(
                f"requests={len(latencies)} p50={p50:.1f}ms p95={p95:.1f}ms "
                f"min={ordered[0]:.1f}ms max={ordered[-1]:.1f}ms"
            )
        print(f"first action: {actions[0].tolist()}")
        print(f"last action:  {actions[-1].tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FastWAM WebSocket client")
    parser.add_argument("--url", default="ws://127.0.0.1:8765/v1/ws")
    parser.add_argument("--token", default=None)
    parser.add_argument("--image", action="append", default=[], help="camera image path (repeatable)")
    parser.add_argument("--synthetic", action="store_true", help="send generated images instead of files")
    parser.add_argument(
        "--no-handshake",
        action="store_true",
        help="skip the hello frame; token goes in ?token= and prompt in every infer",
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=2,
        help="synthetic camera count when --no-handshake is used",
    )
    parser.add_argument("--state", default=None, help="comma separated raw state")
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sigma-shift", type=float, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--id-prefix", default="req")
    parser.add_argument("--max-size", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--output", default=None, help="write the last response to this JSON file")
    parser.add_argument("--verbose", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
