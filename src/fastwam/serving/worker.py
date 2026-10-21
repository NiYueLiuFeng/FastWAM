"""Single-threaded inference worker.

All CUDA work — including model construction, ``torch.compile`` and CUDA graph
capture — happens on one dedicated thread.  FastAPI handlers submit jobs and
await futures, so concurrent WebSocket connections are serialized on the GPU
without racing each other.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from fastwam.utils.logging_config import get_logger

from .protocol import BUSY, TIMEOUT, ProtocolError

logger = get_logger(__name__)


@dataclass
class _Job:
    request: Any
    prompt: str
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop
    enqueued_at: float = field(default_factory=time.monotonic)


class InferenceWorker:
    def __init__(
        self,
        engine_factory: Callable[[], Any],
        *,
        max_queue: int = 16,
        name: str = "fastwam-inference",
    ):
        self._engine_factory = engine_factory
        self._queue: "queue.Queue[_Job | None]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._ready = threading.Event()
        self.init_error: BaseException | None = None
        self.engine = None
        self.max_queue = max_queue

    # --------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._thread.start()

    def wait_ready(self, timeout: float) -> None:
        if not self._ready.wait(timeout):
            raise TimeoutError(f"engine initialization did not finish within {timeout:.1f}s")
        if self.init_error is not None:
            raise RuntimeError("engine initialization failed") from self.init_error

    def stop(self, timeout: float = 30.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("Inference thread did not stop within %.1fs", timeout)

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self.init_error is None and self.engine is not None

    # ------------------------------------------------------------------ thread

    def _run(self) -> None:
        try:
            logger.info("Initializing inference engine on the worker thread")
            self.engine = self._engine_factory()
            warmup = getattr(self.engine, "warmup", None)
            if callable(warmup):
                warmup()
            logger.info("Inference engine is ready")
        except BaseException as exc:  # noqa: BLE001 - surfaced to the server owner
            self.init_error = exc
            logger.exception("Failed to initialize inference engine")
        finally:
            self._ready.set()
        if self.init_error is not None:
            return

        while True:
            job = self._queue.get()
            if job is None:
                break
            self._handle(job)
        logger.info("Inference worker stopped")

    def _handle(self, job: _Job) -> None:
        queue_ms = (time.monotonic() - job.enqueued_at) * 1000.0
        try:
            result = self.engine.infer(job.request, job.prompt)
            result.timing["queue_ms"] = queue_ms
        except BaseException as exc:  # noqa: BLE001 - forwarded to the caller
            self._settle(job, exc=exc)
        else:
            self._settle(job, result=result)

    @staticmethod
    def _settle(job: _Job, *, result=None, exc: BaseException | None = None) -> None:
        def _callback() -> None:
            if job.future.cancelled():
                return
            if exc is not None:
                job.future.set_exception(exc)
            else:
                job.future.set_result(result)

        try:
            job.loop.call_soon_threadsafe(_callback)
        except RuntimeError:
            # The event loop was closed while the job was in flight.
            pass

    # ---------------------------------------------------------------- submit

    async def submit(self, request, prompt: str, timeout_s: float):
        if self._queue.qsize() >= self.max_queue:
            raise ProtocolError(
                BUSY,
                "inference queue is full",
                f"queue_depth={self._queue.qsize()} max_queue={self.max_queue}",
            )
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._queue.put(_Job(request=request, prompt=prompt, future=future, loop=loop))
        try:
            return await asyncio.wait_for(future, timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            raise ProtocolError(
                TIMEOUT,
                "inference request timed out",
                f"timeout_ms={int(timeout_s * 1000)} queue_depth={self._queue.qsize()}",
            ) from exc
