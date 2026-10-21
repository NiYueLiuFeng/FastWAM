"""Entry point helpers: build the FastAPI app and run uvicorn."""

from __future__ import annotations

import logging

import uvicorn
from omegaconf import DictConfig

from fastwam.utils.logging_config import get_logger, setup_logging

from .app import create_app
from .config import ServeConfig
from .engine import InferenceEngine

logger = get_logger(__name__)


def build_app(cfg: DictConfig):
    """Return ``(app, serve_config)`` without starting the server."""
    serve_config = ServeConfig.from_hydra(cfg)

    def engine_factory() -> InferenceEngine:
        return InferenceEngine(cfg, serve_config)

    app = create_app(serve_config, engine_factory)
    return app, serve_config


def run_service(cfg: DictConfig) -> None:
    setup_logging(log_level=getattr(logging, str(cfg.serving.get("log_level", "INFO")).upper()))
    app, serve_config = build_app(cfg)

    uvicorn_config = uvicorn.Config(
        app,
        host=serve_config.host,
        port=serve_config.port,
        log_level=serve_config.log_level.lower(),
        ws_max_size=serve_config.max_message_bytes,
        ws_ping_interval=20.0,
        ws_ping_timeout=20.0,
        timeout_graceful_shutdown=30,
        ssl_certfile=serve_config.tls_certfile,
        ssl_keyfile=serve_config.tls_keyfile,
    )
    scheme = "wss" if serve_config.tls_certfile else "ws"
    logger.info(
        "Starting FastWAM service on %s://%s:%d/v1/ws (device=%s, dry_run=%s)",
        scheme,
        serve_config.host,
        serve_config.port,
        serve_config.device,
        serve_config.dry_run,
    )
    uvicorn.Server(uvicorn_config).run()
