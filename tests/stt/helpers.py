"""Shared helpers for the STT adapter tests.

``FakeServer`` stands in for a provider's HTTP API on ``127.0.0.1`` so the
real vendor SDKs can be exercised end to end without network access or API
keys.  ``Heartbeat`` proves that an ``await`` leaves the event loop free.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# (method, path, headers, body) -> (status, response headers, payload)
Handler = Callable[[str, str, dict[str, str], bytes], tuple[int, dict[str, str], bytes]]


def json_response(
    obj: Any, status: int = 200
) -> tuple[int, dict[str, str], bytes]:
    """Build a JSON reply for a ``FakeServer`` handler."""
    return status, {"content-type": "application/json"}, json.dumps(obj).encode()


class FakeServer:
    """A threaded HTTP server bound to ``127.0.0.1`` only.

    Every request is recorded in :attr:`requests` as
    ``(method, path, headers, body)`` before ``handler`` answers it.
    """

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[tuple[str, str, dict[str, str], bytes]] = []
        outer = self

        class _Request(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args: Any) -> None:
                pass

            def _serve(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length) if length else b""
                headers = {k.lower(): v for k, v in self.headers.items()}
                outer.requests.append((self.command, self.path, headers, body))
                status, reply_headers, payload = outer.handler(
                    self.command, self.path, headers, body
                )
                self.send_response(status)
                for key, value in reply_headers.items():
                    self.send_header(key, value)
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_POST = do_PUT = do_DELETE = _serve

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Request)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> FakeServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class Heartbeat:
    """Counts how often a 20 ms ticker runs on the event loop.

    A call that blocks the loop leaves :meth:`ticks_during` at ~0; one that
    keeps it free ticks about ``duration / 0.02`` times.
    """

    def __init__(self) -> None:
        self._ticks = 0
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(0.02)
            self._ticks += 1

    async def __aenter__(self) -> Heartbeat:
        self._task = asyncio.create_task(self._run())
        await asyncio.sleep(0.1)  # let the ticker get going
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._task is not None
        self._task.cancel()

    async def ticks_during(self, awaitable: Any) -> tuple[Any, int]:
        """Await *awaitable* and return ``(its result, ticks that ran meanwhile)``."""
        before = self._ticks
        result = await awaitable
        return result, self._ticks - before


def write_wav(path: Path, seconds: float = 0.2) -> Path:
    """Write a silent 16 kHz mono WAV file and return *path*."""
    import wave

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


def slow(reply: tuple[int, dict[str, str], bytes], seconds: float) -> Handler:
    """A ``FakeServer`` handler that answers *reply* after *seconds*."""

    def handler(*_: object) -> tuple[int, dict[str, str], bytes]:
        time.sleep(seconds)
        return reply

    return handler
