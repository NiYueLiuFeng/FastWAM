"""Service configuration parsed from the hydra ``serving`` block."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


@dataclass
class ServeConfig:
    device: str = "cuda"
    dtype: str = "bf16"
    ckpt: str | None = None
    dataset_stats_path: str | None = None

    host: str = "0.0.0.0"
    port: int = 8765

    compile_action_infer: bool = True
    force_serve_overrides: bool = True
    text_encoder_device: str = "cuda"
    invert_action_transforms: bool = False

    prompt_cache_size: int = 64
    num_inference_steps: int = 10
    sigma_shift: float | None = None
    action_horizon: int | None = None
    num_video_frames: int | None = None
    action_infer_mode: str = "first_frame"

    max_queue: int = 16
    request_timeout_ms: int = 5000
    init_timeout_s: float = 1800.0
    max_message_bytes: int = 16 * 1024 * 1024
    max_images: int = 8
    max_image_bytes: int = 4 * 1024 * 1024

    warmup: bool = True
    warmup_runs: int = 2
    dry_run: bool = False

    auth_token_env: str = "FASTWAM_SERVICE_TOKEN"
    log_level: str = "INFO"

    # Optional in-process TLS (wss://). Prefer terminating TLS at a reverse proxy.
    tls_certfile: str | None = None
    tls_keyfile: str | None = None

    token: str | None = field(default=None, repr=False)

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "ServeConfig":
        if "serving" not in cfg:
            raise ValueError("Missing `serving` block in the config.")
        raw = OmegaConf.to_container(cfg.serving, resolve=True)
        if not isinstance(raw, dict):
            raise ValueError("`serving` must resolve to a mapping.")
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown `serving` keys: {unknown}")
        sc = cls(**raw)
        sc.validate()
        return sc

    def validate(self) -> None:
        if self.dtype not in _DTYPE_MAP:
            raise ValueError(f"`serving.dtype` must be one of {sorted(_DTYPE_MAP)}, got {self.dtype!r}")
        if self.device not in {"cuda", "cpu"} and not self.device.startswith("cuda:"):
            raise ValueError(f"`serving.device` must be cuda[:N] or cpu, got {self.device!r}")
        if self.text_encoder_device not in {"cuda", "cpu"} and not self.text_encoder_device.startswith("cuda:"):
            raise ValueError(
                "`serving.text_encoder_device` must be cuda[:N] or cpu, "
                f"got {self.text_encoder_device!r}"
            )
        if self.warmup_runs < 1:
            raise ValueError("`serving.warmup_runs` must be >= 1")
        if self.max_queue < 1:
            raise ValueError("`serving.max_queue` must be >= 1")
        if self.action_infer_mode not in {"first_frame", "idm"}:
            raise ValueError("`serving.action_infer_mode` must be first_frame or idm")
        if (self.tls_certfile is None) != (self.tls_keyfile is None):
            raise ValueError("`serving.tls_certfile` and `serving.tls_keyfile` must be set together")
        if not self.dry_run:
            missing = [
                name
                for name in ("ckpt", "dataset_stats_path")
                if getattr(self, name) in (None, "")
            ]
            if missing:
                raise ValueError(
                    f"`serving.{missing[0]}` is required unless `serving.dry_run=true`."
                )
            for name in ("ckpt", "dataset_stats_path"):
                path = Path(getattr(self, name))
                if not path.exists():
                    raise FileNotFoundError(f"`serving.{name}` does not exist: {path}")
        self.token = os.environ.get(self.auth_token_env) or None

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPE_MAP[self.dtype]

    @property
    def request_timeout_s(self) -> float:
        return self.request_timeout_ms / 1000.0
