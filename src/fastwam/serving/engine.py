"""Model loading, preprocessing and action postprocessing for the service."""

from __future__ import annotations

import inspect
import time
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.logging_config import get_logger

from .config import ServeConfig
from .protocol import BAD_REQUEST, INTERNAL, ProtocolError

logger = get_logger(__name__)


@dataclass
class InferResult:
    actions: np.ndarray  # [action_horizon, action_dim], float32, dataset action space
    timing: dict[str, float]
    seed_used: int | None


def center_crop_resize(img: Image.Image, width: int, height: int) -> np.ndarray:
    """Aspect-preserving resize + center crop (same as the LIBERO eval script)."""
    src_w, src_h = img.size
    scale = max(width / src_w, height / src_h)
    resized = img.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left, top = max((rw - width) // 2, 0), max((rh - height) // 2, 0)
    return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8)


class InferenceEngine:
    """Loads one FastWAM checkpoint and serves action chunks.

    All methods must be called from the same thread (see ``worker.py``) because
    ``torch.compile`` / CUDA graph state is cached on the model.
    """

    def __init__(self, cfg: DictConfig, sc: ServeConfig):
        self.cfg = cfg
        self.sc = sc
        # Dry-run touches no model, so it stays on CPU: this keeps the protocol
        # check usable on hosts whose GPU the installed torch does not support.
        self.device = "cpu" if sc.dry_run else sc.device
        self.dtype = torch.float32 if sc.dry_run else sc.torch_dtype
        self._prompt_cache: "OrderedDict[str, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()

        self.model = None
        self.processor = self._build_processor()
        self.shape_meta = self.processor.shape_meta

        self.cameras = [
            {"key": str(meta["key"]), "shape": tuple(int(x) for x in meta["shape"])}
            for meta in self.shape_meta["images"]
        ]
        self.action_keys = [str(meta["key"]) for meta in self.shape_meta["action"]]
        self.state_keys = [str(meta["key"]) for meta in self.shape_meta["state"]]
        self.state_dim = sum(int(meta["raw_shape"]) for meta in self.shape_meta["state"])
        self.action_dim = int(self.processor.action_output_dim)
        # The model emits one flat vector; splitting it back per action key
        # requires the configured `shape` values to add up to `action_output_dim`.
        action_shape_sum = sum(int(meta["shape"]) for meta in self.shape_meta["action"])
        if action_shape_sum != self.action_dim:
            raise ValueError(
                "sum(shape_meta.action[*].shape) != processor.action_output_dim: "
                f"{action_shape_sum} != {self.action_dim}. Check the data config's "
                "action transforms."
            )

        self.video_size = tuple(int(x) for x in cfg.data.train.video_size)
        self.concat_multi_camera = str(cfg.data.train.get("concat_multi_camera", "horizontal"))
        self.num_frames = int(cfg.data.train.num_frames)
        self.action_horizon = (
            int(sc.action_horizon) if sc.action_horizon is not None else self.num_frames - 1
        )
        freq_ratio = int(cfg.data.train.get("action_video_freq_ratio", 4))
        self.num_video_frames = (
            int(sc.num_video_frames)
            if sc.num_video_frames is not None
            else (self.num_frames - 1) // freq_ratio + 1
        )

        if not sc.dry_run:
            self.model = self._build_model()
            self.variant = type(self.model).__name__
            self._supports_num_video_frames = (
                "num_video_frames" in inspect.signature(self.model.infer_action).parameters
            )
            self._supports_action_infer_mode = (
                "action_infer_mode" in inspect.signature(self.model.infer_action).parameters
            )
            self.proprio_dim = getattr(self.model, "proprio_dim", None)
        else:
            self.variant = "dry-run"
            self._supports_num_video_frames = False
            self._supports_action_infer_mode = False
            self.proprio_dim = self.state_dim

        self._validate_video_size()

    # ------------------------------------------------------------------ setup

    def _build_processor(self):
        processor_cfg = OmegaConf.create(
            OmegaConf.to_container(self.cfg.data.train.processor, resolve=True)
        )
        processor = instantiate(processor_cfg).eval()
        if not self.sc.dry_run:
            stats = load_dataset_stats_from_json(self.sc.dataset_stats_path)
            processor.set_normalizer_from_stats(stats)
        return processor

    def _build_model(self):
        model_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg.model, resolve=True))
        if self.sc.force_serve_overrides:
            model_cfg.load_text_encoder = True
            model_cfg.skip_dit_load_from_pretrain = True
            model_cfg.action_dit_pretrained_path = None
        model = instantiate(model_cfg, model_dtype=self.dtype, device=self.device)
        model.load_checkpoint(self.sc.ckpt)
        model = model.to(self.device).eval()

        text_encoder = getattr(model, "text_encoder", None)
        if text_encoder is not None and self.sc.text_encoder_device != self.device:
            logger.info(
                "Moving text encoder from %s to %s",
                next(text_encoder.parameters()).device,
                self.sc.text_encoder_device,
            )
            text_encoder.to(self.sc.text_encoder_device)
        return model

    def _validate_video_size(self) -> None:
        height, width = self.video_size
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`video_size` {self.video_size} must be multiples of 16 "
                "(it is VAE-encoded by the model)."
            )

        if self.concat_multi_camera == "robotwin":
            expected = (384, 320)
        elif not self.cameras:
            expected = (0, 0)
        else:
            shapes = [c["shape"] for c in self.cameras]
            if self.concat_multi_camera == "horizontal":
                expected = (shapes[0][1], sum(shape[2] for shape in shapes))
            elif self.concat_multi_camera == "vertical":
                expected = (sum(shape[1] for shape in shapes), shapes[0][2])
            else:
                raise ValueError(
                    f"Unsupported `concat_multi_camera` {self.concat_multi_camera!r}; "
                    "expected horizontal, vertical or robotwin."
                )

        if tuple(expected) != (height, width):
            raise ValueError(
                "`video_size` does not match the concatenated camera shapes: "
                f"video_size={(height, width)}, expected={tuple(expected)} "
                f"(concat_multi_camera={self.concat_multi_camera!r}, "
                f"cameras={[c['shape'] for c in self.cameras]})."
            )

    # -------------------------------------------------------------- metadata

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "action_horizon": self.action_horizon,
            "video_size": list(self.video_size),
            "concat_multi_camera": self.concat_multi_camera,
            "cameras": [{"key": c["key"], "shape": list(c["shape"])} for c in self.cameras],
            "action_keys": self.action_keys,
            "state_keys": self.state_keys,
            "normalization": str(getattr(self.processor, "norm_default_mode", "unknown")),
            "uses_proprio": self.proprio_dim is not None,
            "dry_run": bool(self.sc.dry_run),
            "compile_action_infer": bool(self.sc.compile_action_infer),
        }

    @property
    def limits(self) -> dict[str, Any]:
        return {
            "max_images": self.sc.max_images,
            "max_image_bytes": self.sc.max_image_bytes,
            "max_state_dim": self.state_dim,
            "queue_depth": self.sc.max_queue,
            "request_timeout_ms": self.sc.request_timeout_ms,
        }

    # ------------------------------------------------------------ preprocessing

    def _decode_image(self, payload) -> Image.Image:
        if len(payload.data) > self.sc.max_image_bytes:
            raise ProtocolError(
                BAD_REQUEST,
                f"image exceeds max_image_bytes={self.sc.max_image_bytes}",
                f"got {len(payload.data)} bytes",
            )
        try:
            img = Image.open(BytesIO(payload.data))
            img.load()
        except Exception as exc:
            raise ProtocolError(BAD_REQUEST, "failed to decode image", str(exc)) from exc
        if img.format not in {"JPEG", "PNG"}:
            raise ProtocolError(
                BAD_REQUEST, f"unsupported image format {img.format!r}; use jpeg or png"
            )
        return img.convert("RGB")

    def _order_images(self, images) -> list[Image.Image]:
        if len(images) != len(self.cameras):
            raise ProtocolError(
                BAD_REQUEST,
                f"expected {len(self.cameras)} images, got {len(images)}",
                f"camera order: {[c['key'] for c in self.cameras]}",
            )
        if len(images) > self.sc.max_images:
            raise ProtocolError(BAD_REQUEST, f"at most {self.sc.max_images} images are supported")

        keys = [img.key for img in images]
        if any(key is not None for key in keys) and not all(key is not None for key in keys):
            raise ProtocolError(
                BAD_REQUEST, "either set `key` on every image or on none of them"
            )
        if keys and keys[0] is not None:
            by_key = {img.key: img for img in images}
            expected = [c["key"] for c in self.cameras]
            missing = [key for key in expected if key not in by_key]
            if missing:
                raise ProtocolError(BAD_REQUEST, f"missing camera images: {missing}")
            return [self._decode_image(by_key[key]) for key in expected]
        return [self._decode_image(img) for img in images]

    def _preprocess_images(self, images) -> torch.Tensor:
        decoded = self._order_images(images)

        if self.concat_multi_camera == "robotwin":
            if len(decoded) != 3:
                raise ProtocolError(
                    BAD_REQUEST, "concat_multi_camera='robotwin' requires exactly 3 cameras"
                )
            top = np.asarray(decoded[0].resize((320, 256), resample=Image.BILINEAR), dtype=np.uint8)
            left = np.asarray(decoded[1].resize((160, 128), resample=Image.BILINEAR), dtype=np.uint8)
            right = np.asarray(decoded[2].resize((160, 128), resample=Image.BILINEAR), dtype=np.uint8)
            rgb = np.concatenate([top, np.concatenate([left, right], axis=1)], axis=0)
        else:
            frames = [
                center_crop_resize(img, width=shape[2], height=shape[1])
                for img, shape in zip(decoded, [c["shape"] for c in self.cameras])
            ]
            if len(frames) == 1:
                rgb = frames[0]
            elif self.concat_multi_camera == "horizontal":
                rgb = np.concatenate(frames, axis=1)
            elif self.concat_multi_camera == "vertical":
                rgb = np.concatenate(frames, axis=0)
            else:
                raise ProtocolError(
                    INTERNAL,
                    f"unsupported concat_multi_camera={self.concat_multi_camera!r}",
                )

        expected_hw = tuple(self.video_size)
        if rgb.shape[:2] != expected_hw:
            raise ProtocolError(
                INTERNAL,
                f"preprocessed image is {rgb.shape[:2]}, expected video_size {expected_hw}",
                "check shape_meta / video_size / concat_multi_camera in the data config",
            )

        x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        return x * (2.0 / 255.0) - 1.0

    def _normalize_state(self, state: list[float]) -> torch.Tensor | None:
        if self.proprio_dim is None:
            return None
        if len(state) != self.state_dim:
            raise ProtocolError(
                BAD_REQUEST,
                f"state must have {self.state_dim} values, got {len(state)}",
                f"state_keys: {self.state_keys}",
            )
        arr = np.asarray(state, dtype=np.float32)

        if self.sc.dry_run:
            return torch.as_tensor(arr).unsqueeze(0).to(self.device)

        batch: dict[str, dict[str, torch.Tensor]] = {"state": {}}
        offset = 0
        for meta in self.shape_meta["state"]:
            raw_dim = int(meta["raw_shape"])
            key = str(meta["key"])
            batch["state"][key] = torch.as_tensor(
                arr[offset : offset + raw_dim], dtype=torch.float32
            ).unsqueeze(0)
            offset += raw_dim
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        merged = torch.cat([batch["state"][str(m["key"])] for m in self.shape_meta["state"]], dim=-1)
        if self.proprio_dim is not None and merged.shape[-1] != self.proprio_dim:
            raise ProtocolError(
                INTERNAL,
                f"normalized state has {merged.shape[-1]} dims, model expects {self.proprio_dim}",
                "check shape_meta.state and the processor's action_state_transforms",
            )
        return merged.to(device=self.device, dtype=self.dtype)

    # ----------------------------------------------------------- postprocessing

    def _denormalize_action(
        self, action: torch.Tensor, state: list[float] | None = None
    ) -> np.ndarray:
        """``action`` is [T, action_dim] on CPU, still normalized.

        When the processor carries action transforms (e.g. delta actions), they
        are inverted with the *raw* ``state`` so the caller receives absolute
        actions, not offsets.
        """
        if self.sc.dry_run:
            return action.to(torch.float32).numpy()

        x = action.to(torch.float32).unsqueeze(0)
        parts = []
        offset = 0
        for meta in self.shape_meta["action"]:
            dim = int(meta["shape"])
            key = str(meta["key"])
            normalizer = self.processor.normalizer.normalizers["action"][key]
            parts.append(normalizer.backward(x[:, :, offset : offset + dim]))
            offset += dim
        merged = torch.cat(parts, dim=-1)[0]
        if state is not None and self.processor.action_state_transforms:
            merged = self._invert_action_transforms(merged, state)
        return merged.numpy()

    def _invert_action_transforms(
        self, action: torch.Tensor, state: list[float]
    ) -> torch.Tensor:
        arr = np.asarray(state, dtype=np.float32)
        batch: dict[str, dict[str, torch.Tensor]] = {"action": {}, "state": {}}
        offset = 0
        for meta in self.shape_meta["action"]:
            dim = int(meta["shape"])
            key = str(meta["key"])
            batch["action"][key] = action[:, offset : offset + dim].unsqueeze(0)
            offset += dim
        offset = 0
        for meta in self.shape_meta["state"]:
            dim = int(meta["raw_shape"])
            key = str(meta["key"])
            batch["state"][key] = torch.as_tensor(
                arr[offset : offset + dim], dtype=torch.float32
            ).unsqueeze(0)
            offset += dim
        for trans in reversed(self.processor.action_state_transforms):
            batch = trans.backward(batch)
        return torch.cat(
            [batch["action"][str(m["key"])][0] for m in self.shape_meta["action"]],
            dim=-1,
        )

    # ------------------------------------------------------------------ prompt

    def _get_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor, float]:
        cached = self._prompt_cache.get(prompt)
        if cached is not None:
            self._prompt_cache.move_to_end(prompt)
            return cached[0], cached[1], 0.0

        start = time.perf_counter()
        text = DEFAULT_PROMPT.format(task=prompt)
        context, context_mask = self._encode_text(text)
        context = context.detach()
        context_mask = context_mask.detach()

        self._prompt_cache[prompt] = (context, context_mask)
        if len(self._prompt_cache) > self.sc.prompt_cache_size:
            self._prompt_cache.popitem(last=False)
        return context, context_mask, (time.perf_counter() - start) * 1000.0

    def _encode_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sc.dry_run:
            context = torch.zeros(
                (1, 128, int(self.cfg.model.video_dit_config.text_dim)),
                device=self.device,
                dtype=self.dtype,
            )
            return context, torch.ones((1, 128), dtype=torch.bool, device=self.device)

        text_encoder = getattr(self.model, "text_encoder", None)
        if text_encoder is None:
            raise ProtocolError(
                INTERNAL,
                "text encoder is not loaded; set model.load_text_encoder=true",
            )
        encoder_device = next(text_encoder.parameters()).device
        if encoder_device.type == "cpu" and self.device != "cpu":
            ids, mask = self.model.tokenizer(text, return_mask=True, add_special_tokens=True)
            emb = text_encoder(ids, mask)
            seq_lens = mask.gt(0).sum(dim=1).long()
            for i, length in enumerate(seq_lens):
                emb[i, length:] = 0
            context = emb.to(device=self.device, dtype=self.dtype)
            return context, torch.ones_like(mask, device=self.device)
        return self.model.encode_prompt(text)

    # ------------------------------------------------------------------ infer

    def warmup(self) -> dict[str, float]:
        if self.sc.dry_run:
            return {"warmup_ms": 0.0}
        height, width = self.video_size
        image = torch.zeros((1, 3, height, width), device=self.device, dtype=self.dtype)
        proprio = (
            torch.zeros((1, int(self.proprio_dim)), device=self.device, dtype=self.dtype)
            if self.proprio_dim is not None
            else None
        )
        context, context_mask, _ = self._get_context("warmup task")
        timings = []
        for _ in range(self.sc.warmup_runs):
            start = time.perf_counter()
            self._predict(image, proprio, context, context_mask, self._default_options())
            self._synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
        logger.info("Warmup finished: %s", [f"{t:.1f} ms" for t in timings])
        return {"warmup_ms": float(timings[-1])}

    def _synchronize(self) -> None:
        if not self.sc.dry_run and torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)

    def _default_options(self):
        return {
            "num_inference_steps": self.sc.num_inference_steps,
            "sigma_shift": self.sc.sigma_shift,
            "seed": None,
            "action_horizon": self.action_horizon,
        }

    def infer(self, request, prompt: str) -> InferResult:
        total_start = time.perf_counter()

        pre_start = time.perf_counter()
        image = self._preprocess_images(request.images)
        proprio = self._normalize_state(request.state)
        preprocess_ms = (time.perf_counter() - pre_start) * 1000.0

        context, context_mask, text_ms = self._get_context(prompt)

        options = request.options
        opts = self._default_options()
        if options.num_inference_steps is not None:
            opts["num_inference_steps"] = options.num_inference_steps
        if options.sigma_shift is not None:
            opts["sigma_shift"] = options.sigma_shift
        if options.seed is not None:
            opts["seed"] = options.seed
        if options.action_horizon is not None:
            opts["action_horizon"] = options.action_horizon

        infer_start = time.perf_counter()
        action = self._predict(image, proprio, context, context_mask, opts)
        self._synchronize()
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        post_start = time.perf_counter()
        actions = self._denormalize_action(action, request.state)
        postprocess_ms = (time.perf_counter() - post_start) * 1000.0

        timing = {
            "preprocess_ms": preprocess_ms,
            "text_ms": text_ms,
            "infer_ms": infer_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": (time.perf_counter() - total_start) * 1000.0,
        }
        return InferResult(actions=actions, timing=timing, seed_used=opts["seed"])

    def _predict(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor | None,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        opts: dict[str, Any],
    ) -> torch.Tensor:
        if self.sc.dry_run:
            horizon = int(opts["action_horizon"])
            steps = np.arange(horizon * self.action_dim, dtype=np.float32) / (self.action_dim * 4.0)
            return torch.from_numpy(
                (0.1 * np.sin(steps)).reshape(horizon, self.action_dim).copy()
            )

        kwargs: dict[str, Any] = dict(
            prompt=None,
            input_image=image,
            action_horizon=int(opts["action_horizon"]),
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt="",
            text_cfg_scale=1.0,
            num_inference_steps=int(opts["num_inference_steps"]),
            sigma_shift=opts["sigma_shift"],
            seed=opts["seed"],
            rand_device="cpu",
            tiled=False,
            compile_action_infer=bool(self.sc.compile_action_infer),
        )
        if self._supports_num_video_frames:
            kwargs["num_video_frames"] = self.num_video_frames
        if self._supports_action_infer_mode:
            kwargs["action_infer_mode"] = self.sc.action_infer_mode

        out = self.model.infer_action(**kwargs)
        return out["action"]
