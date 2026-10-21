"""FastWAM cloud inference service (WebSocket + msgpack).

The package is optional: install it with ``pip install -e .[serving]``.
"""

from .protocol import PROTOCOL_VERSION

__all__ = ["PROTOCOL_VERSION"]
