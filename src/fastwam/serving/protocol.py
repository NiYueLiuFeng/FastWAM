"""Wire protocol for the FastWAM WebSocket inference service.

Every WebSocket binary frame carries exactly one msgpack map.  Requests are
validated with pydantic; responses are plain dicts converted with
``msgpack.packb``.  Only map/array/str/int/float/bool/bin are used so that
clients in other languages can decode the frames with any msgpack library.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Optional, Union

try:
    import msgpack
except ImportError as exc:  # pragma: no cover - depends on optional extra
    raise ImportError(
        "The FastWAM serving extra is not installed. "
        "Run `pip install -e .[serving]` (or install fastapi/uvicorn/msgpack/websockets)."
    ) from exc

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROTOCOL_VERSION = 1

# Error codes shared by the client and the server.
BAD_REQUEST = "BAD_REQUEST"
UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
UNAUTHORIZED = "UNAUTHORIZED"
BUSY = "BUSY"
TIMEOUT = "TIMEOUT"
INTERNAL = "INTERNAL"


class ProtocolError(Exception):
    """A request-level failure that maps to an ``error`` frame."""

    def __init__(self, code: str, message: str, detail: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CameraSpec(_Model):
    key: str
    shape: tuple[int, int, int]  # [C, H, W] after preprocessing


class ImagePayload(_Model):
    key: Optional[str] = None
    format: Literal["jpeg", "jpg", "png"] = "jpeg"
    data: bytes = Field(min_length=1)

    @field_validator("format")
    @classmethod
    def _normalize_format(cls, value: str) -> str:
        return "jpeg" if value == "jpg" else value


class InferOptions(_Model):
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=200)
    sigma_shift: Optional[float] = Field(default=None, gt=0.0, le=100.0)
    seed: Optional[int] = Field(default=None, ge=0, le=2**63 - 1)
    action_horizon: Optional[int] = Field(default=None, ge=1, le=1024)
    timeout_ms: Optional[int] = Field(default=None, ge=1, le=600_000)


class HelloRequest(_Model):
    v: int = PROTOCOL_VERSION
    type: Literal["hello"]
    token: Optional[str] = None
    client: Optional[str] = None
    prompt: Optional[str] = None
    options: InferOptions = Field(default_factory=InferOptions)

    @field_validator("v")
    @classmethod
    def _check_version(cls, value: int) -> int:
        if value != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version {value}; expected {PROTOCOL_VERSION}")
        return value


class InferRequest(_Model):
    v: int = PROTOCOL_VERSION
    type: Literal["infer"]
    id: Union[str, int]
    prompt: Optional[str] = None
    images: list[ImagePayload] = Field(min_length=1)
    state: list[float] = Field(min_length=1)
    options: InferOptions = Field(default_factory=InferOptions)

    @field_validator("v")
    @classmethod
    def _check_version(cls, value: int) -> int:
        if value != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version {value}; expected {PROTOCOL_VERSION}")
        return value


def is_hello(payload: dict[str, Any]) -> bool:
    return payload.get("type") == "hello"


def pack(payload: dict[str, Any]) -> bytes:
    return msgpack.packb(payload, use_bin_type=True)


def unpack(data: bytes) -> dict[str, Any]:
    try:
        payload = msgpack.unpackb(data, raw=False, strict_map_key=False)
    except Exception as exc:  # msgpack raises several exception types
        raise ProtocolError(BAD_REQUEST, "failed to decode msgpack frame", str(exc)) from exc
    if not isinstance(payload, dict):
        raise ProtocolError(BAD_REQUEST, "frame must be a msgpack map")
    return payload


def parse_hello(payload: dict[str, Any]) -> HelloRequest:
    try:
        return HelloRequest.model_validate(payload)
    except Exception as exc:
        raise ProtocolError(BAD_REQUEST, "invalid hello frame", str(exc)) from exc


def parse_infer(payload: dict[str, Any]) -> InferRequest:
    try:
        return InferRequest.model_validate(payload)
    except Exception as exc:
        raise ProtocolError(BAD_REQUEST, "invalid infer frame", str(exc)) from exc


def hello_ack(
    *,
    server: str,
    session_id: str,
    model_info: dict[str, Any],
    limits: dict[str, Any],
) -> dict[str, Any]:
    return {
        "v": PROTOCOL_VERSION,
        "type": "hello_ack",
        "server": server,
        "session_id": session_id,
        "model": model_info,
        "limits": limits,
    }


def actions_ack(
    *,
    request_id: Union[str, int],
    actions: list[list[float]],
    action_keys: list[str],
    timing: dict[str, float],
    seed_used: Optional[int],
) -> dict[str, Any]:
    return {
        "v": PROTOCOL_VERSION,
        "type": "actions",
        "id": request_id,
        "actions": actions,
        "action_horizon": len(actions),
        "action_dim": len(actions[0]) if actions else 0,
        "action_keys": action_keys,
        "space": {"state": "dataset observation.state", "action": "dataset action"},
        "seed_used": seed_used,
        "timing": timing,
        "server_ts": time.time(),
    }


def error_frame(
    *,
    request_id: Union[str, int, None],
    code: str,
    message: str,
    detail: str | None = None,
) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "v": PROTOCOL_VERSION,
        "type": "error",
        "id": request_id,
        "code": code,
        "message": message,
    }
    if detail:
        frame["detail"] = detail
    return frame
