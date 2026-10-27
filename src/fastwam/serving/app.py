"""FastAPI application exposing FastWAM over WebSocket + msgpack."""

from __future__ import annotations

import asyncio
import hmac
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from fastwam.utils.logging_config import get_logger

from .config import ServeConfig
from .protocol import (
    BAD_REQUEST,
    INTERNAL,
    UNAUTHORIZED,
    UNSUPPORTED_FORMAT,
    ProtocolError,
    actions_ack,
    error_frame,
    hello_ack,
    pack,
    parse_hello,
    parse_infer,
    unpack,
)
from .worker import InferenceWorker

logger = get_logger(__name__)

SERVER_NAME = "fastwam/0.1.0"
_WS_UNAUTHORIZED = 4401
_WS_MESSAGE_TOO_BIG = 1009


class SessionState:
    def __init__(self) -> None:
        self.id = uuid.uuid4().hex
        self.client: str | None = None
        self.prompt: str | None = None
        self.options: dict[str, Any] = {}


def _authorized(sc: ServeConfig, provided: str | None) -> bool:
    if not sc.token:
        return True
    if not provided:
        return False
    return hmac.compare_digest(provided, sc.token)


def create_app(serve_config: ServeConfig, engine_factory: Callable[[], Any]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker = InferenceWorker(engine_factory, max_queue=serve_config.max_queue)
        app.state.worker = worker
        worker.start()

        if serve_config.token is None:
            logger.warning(
                "No auth token configured (`%s` is unset). The WebSocket endpoint is open; "
                "only do this on a trusted network.",
                serve_config.auth_token_env,
            )

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, worker.wait_ready, serve_config.init_timeout_s)
            app.state.engine = worker.engine
            logger.info("Service is ready")
        except Exception:
            logger.exception("Service started without a usable model; /readyz will return 503")
        try:
            yield
        finally:
            worker.stop()

    app = FastAPI(title="FastWAM Inference Service", version="0.1.0", lifespan=lifespan)
    app.state.serve_config = serve_config
    app.state.engine = None

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        worker: InferenceWorker = app.state.worker
        if worker.ready:
            return {"ready": True}
        detail = None if worker.init_error is None else repr(worker.init_error)
        return JSONResponse(status_code=503, content={"ready": False, "error": detail})

    @app.get("/v1/info")
    async def info():
        engine = app.state.engine
        if engine is None:
            return JSONResponse(status_code=503, content={"ready": False})
        return {
            "server": SERVER_NAME,
            "protocol_version": 1,
            "model": engine.model_info,
            "limits": engine.limits,
        }

    @app.websocket("/v1/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        session = SessionState()
        query_token = websocket.query_params.get("token")
        # A matching `?token=` authenticates the connection up front, so clients
        # that skip `hello` can send `infer` as their first frame.
        authenticated = _authorized(serve_config, query_token)

        async def send(frame: dict[str, Any]) -> None:
            await websocket.send_bytes(pack(frame))

        async def receive_payload() -> dict[str, Any]:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()
            if message.get("text") is not None:
                raise ProtocolError(
                    UNSUPPORTED_FORMAT,
                    "text frames are not supported; send one msgpack map per binary frame",
                )
            data = message.get("bytes") or b""
            if len(data) > serve_config.max_message_bytes:
                await send(
                    error_frame(
                        request_id=None,
                        code=UNSUPPORTED_FORMAT,
                        message=f"frame exceeds max_message_bytes={serve_config.max_message_bytes}",
                    )
                )
                await websocket.close(code=_WS_MESSAGE_TOO_BIG)
                raise WebSocketDisconnect()
            return unpack(data)

        try:
            # `hello` is optional: a client that sends `?token=...` and a
            # `prompt` with every request can start with `infer` directly.
            while True:
                payload = await receive_payload()
                message_type = payload.get("type")
                request_id = payload.get("id")

                if message_type == "hello":
                    try:
                        hello = parse_hello(payload)
                    except ProtocolError as exc:
                        await send(
                            error_frame(
                                request_id=None,
                                code=exc.code,
                                message=exc.message,
                                detail=exc.detail,
                            )
                        )
                        await websocket.close(code=1002)
                        return

                    if not _authorized(serve_config, hello.token or query_token):
                        await send(
                            error_frame(
                                request_id=None,
                                code=UNAUTHORIZED,
                                message="invalid or missing token",
                            )
                        )
                        await websocket.close(code=_WS_UNAUTHORIZED)
                        return
                    authenticated = True

                    session.client = hello.client
                    session.prompt = hello.prompt
                    session.options = {
                        key: value
                        for key, value in hello.options.model_dump().items()
                        if value is not None
                    }
                    logger.info("Session %s connected (client=%s)", session.id, session.client)

                    engine = app.state.engine
                    if engine is None:
                        await send(
                            error_frame(
                                request_id=None,
                                code=INTERNAL,
                                message="model is not ready",
                                detail=repr(app.state.worker.init_error)
                                if app.state.worker.init_error
                                else None,
                            )
                        )
                        await websocket.close(code=1013)
                        return

                    await send(
                        hello_ack(
                            server=SERVER_NAME,
                            session_id=session.id,
                            model_info=engine.model_info,
                            limits=engine.limits,
                        )
                    )
                    continue

                if message_type == "ping":
                    await send({"v": 1, "type": "pong", "ts": payload.get("ts")})
                    continue

                if not authenticated:
                    await send(
                        error_frame(
                            request_id=request_id,
                            code=UNAUTHORIZED,
                            message="invalid or missing token",
                            detail="add ?token=... to the URL or send a hello frame",
                        )
                    )
                    await websocket.close(code=_WS_UNAUTHORIZED)
                    return

                if message_type == "reset":
                    session.prompt = None
                    session.options = {}
                    await send({"v": 1, "type": "reset_ack", "session_id": session.id})
                    continue
                if message_type != "infer":
                    await send(
                        error_frame(
                            request_id=request_id,
                            code=UNSUPPORTED_FORMAT,
                            message=f"unknown message type {message_type!r}",
                        )
                    )
                    continue

                engine = app.state.engine
                if engine is None:
                    await send(
                        error_frame(
                            request_id=request_id,
                            code=INTERNAL,
                            message="model is not ready",
                            detail=repr(app.state.worker.init_error)
                            if app.state.worker.init_error
                            else None,
                        )
                    )
                    await websocket.close(code=1013)
                    return

                try:
                    request = parse_infer(payload)
                    prompt = request.prompt or session.prompt
                    if not prompt:
                        raise ProtocolError(
                            BAD_REQUEST,
                            "missing prompt",
                            "send `prompt` in the hello frame or in every infer request",
                        )
                    request.options = _merge_options(session.options, request.options)
                    timeout_s = (
                        request.options.timeout_ms / 1000.0
                        if request.options.timeout_ms is not None
                        else serve_config.request_timeout_s
                    )
                    result = await app.state.worker.submit(request, prompt, timeout_s)
                except ProtocolError as exc:
                    await send(
                        error_frame(
                            request_id=request_id,
                            code=exc.code,
                            message=exc.message,
                            detail=exc.detail,
                        )
                    )
                    continue
                except WebSocketDisconnect:
                    raise
                except Exception as exc:  # noqa: BLE001 - keep the connection alive
                    logger.exception("inference failed (id=%s)", request_id)
                    await send(
                        error_frame(
                            request_id=request_id,
                            code=INTERNAL,
                            message="inference failed",
                            detail=str(exc),
                        )
                    )
                    continue

                await send(
                    actions_ack(
                        request_id=request.id,
                        actions=result.actions.tolist(),
                        action_keys=engine.action_keys,
                        timing=result.timing,
                        seed_used=result.seed_used,
                    )
                )
                logger.info(
                    "session=%s client=%s id=%s steps=%s total_ms=%.1f",
                    session.id,
                    session.client,
                    request.id,
                    request.options.num_inference_steps,
                    result.timing.get("total_ms", 0.0),
                )

        except WebSocketDisconnect:
            logger.info("Session %s disconnected", session.id)
        except ProtocolError as exc:
            try:
                await send(
                    error_frame(request_id=None, code=exc.code, message=exc.message, detail=exc.detail)
                )
            except Exception:  # noqa: BLE001 - the socket may already be gone
                pass
        except Exception:  # noqa: BLE001
            logger.exception("unhandled WebSocket error (session=%s)", session.id)
            try:
                await websocket.close(code=1011)
            except Exception:  # noqa: BLE001
                pass

    return app


def _merge_options(session_options: dict[str, Any], request_options) -> Any:
    """Session defaults are overridden by non-None per-request options."""
    merged = dict(session_options)
    for key, value in request_options.model_dump().items():
        if value is not None:
            merged[key] = value
    return type(request_options)(**merged)
