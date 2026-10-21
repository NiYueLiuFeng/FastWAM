"""Minimal FastWAM inference loop for a custom robot/environment.

This is a template for running a FastWAM checkpoint trained on your own
LeRobot-format dataset. Replace the placeholder functions marked with
"TODO(custom)" with your robot / environment API.

Usage (from the repo root):

    python examples/infer_custom.py \
        task=mydata_uncond_2cam224_1e-4 \
        +ckpt=./runs/mydata_uncond_2cam224_1e-4/<run_id>/checkpoints/weights/<step>.pt \
        +dataset_stats_path=./runs/mydata_uncond_2cam224_1e-4/<run_id>/dataset_stats.json \
        model.load_text_encoder=true \
        model.skip_dit_load_from_pretrain=true \
        model.action_dit_pretrained_path=null \
        +CUSTOM.task_description="put the bowl on the plate" \
        +CUSTOM.replan_steps=10
"""

from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json


# ---------------------------------------------------------------------------
# TODO(custom): replace the three functions below with your robot/env API.
# ---------------------------------------------------------------------------

def get_observation() -> dict[str, Any]:
    """Return the current observation.

    Keys must match your shape_meta order:
      - one entry per camera, in the same order as shape_meta.images,
        e.g. {"image": PIL.Image, "wrist_image": PIL.Image}
      - "state": np.ndarray [state_dim] with the current observation.state
    """
    raise NotImplementedError("Wire this to your robot/environment API.")


def step_env(action: np.ndarray) -> dict[str, Any]:
    """Execute one action [action_dim] and return the next observation."""
    raise NotImplementedError("Wire this to your robot/environment API.")


def done(obs: dict[str, Any]) -> bool:
    """Return True when the episode is finished."""
    raise NotImplementedError("Wire this to your robot/environment API.")


# ---------------------------------------------------------------------------
# Model input construction (mirrors experiments/libero/eval_libero_single.py)
# ---------------------------------------------------------------------------

def center_crop_resize(img: Image.Image, width: int, height: int) -> np.ndarray:
    """Aspect-preserving resize + center crop, same as LIBERO eval."""
    src_w, src_h = img.size
    scale = max(width / src_w, height / src_h)
    resized = img.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left, top = max((rw - width) // 2, 0), max((rh - height) // 2, 0)
    return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8)


def build_model_input(
    cfg: DictConfig,
    processor: FastWAMProcessor,
    cam_images: list[Image.Image],
    proprio: np.ndarray,
    device: str,
    dtype: torch.dtype,
):
    """Resize each camera, concat them, and normalize to [-1, 1]."""
    image_meta = processor.shape_meta["images"]
    frames = []
    for meta, img in zip(image_meta, cam_images):
        _, h, w = meta["shape"]  # [C, H, W]
        frames.append(center_crop_resize(img, width=w, height=h))

    concat = cfg.data.train.get("concat_multi_camera", "horizontal")
    if len(frames) == 1:
        rgb = frames[0]
    elif concat == "horizontal":
        rgb = np.concatenate(frames, axis=1)
    elif concat == "vertical":
        rgb = np.concatenate(frames, axis=0)
    else:
        raise ValueError(f"Unsupported concat_multi_camera for custom eval: {concat}")

    video_size = cfg.data.train.video_size  # [H, W]
    assert rgb.shape[:2] == tuple(video_size), (
        f"Concatenated image {rgb.shape[:2]} != video_size {video_size}"
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0  # [1, 3, H, W]

    # Normalize proprio with the training-time processor
    state_key = processor.shape_meta["state"][0]["key"]
    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    proprio_norm = state_batch["state"][state_key]
    return x, proprio_norm


def denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    action_key = processor.shape_meta["action"][0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    if action.ndim == 2:
        action = action.unsqueeze(0)
    return normalizer.backward(action.to(torch.float32, device="cpu")).numpy()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    device, dtype = "cuda", torch.bfloat16

    # --- 1. Build model & processor ---
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    model.load_checkpoint(cfg.ckpt)
    model = model.to(device).eval()

    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(cfg.dataset_stats_path))

    action_horizon = int(cfg.data.train.num_frames) - 1  # 32 by default
    custom = cfg.get("CUSTOM") or {}
    replan_steps = int(custom.get("replan_steps", 10))
    num_inference_steps = int(custom.get("num_inference_steps", 20))
    sigma_shift = custom.get("sigma_shift", 1.0)
    sigma_shift = None if sigma_shift is None else float(sigma_shift)
    task_description = str(custom.get("task_description", ""))

    # --- 2. Rollout loop ---
    obs = get_observation()
    while not done(obs):
        cam_images = [obs[meta["key"]] for meta in processor.shape_meta["images"]]
        proprio = np.asarray(obs["state"], dtype=np.float32)

        x, proprio_norm = build_model_input(cfg, processor, cam_images, proprio, device, dtype)
        prompt = DEFAULT_PROMPT.format(task=task_description)  # keep the training-time template

        with torch.no_grad():
            pred = model.infer_action(
                prompt=prompt,
                input_image=x,
                proprio=proprio_norm,
                action_horizon=action_horizon,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                negative_prompt="",
                text_cfg_scale=1.0,
            )
        action_chunk = denormalize_action(pred["action"], processor)[0]  # [action_horizon, D]

        for action in action_chunk[:replan_steps]:
            obs = step_env(action)
            if done(obs):
                break


if __name__ == "__main__":
    main()
