"""A tiny fake provider API on 127.0.0.1 for exercising the real SDKs.

The server speaks HTTP/1.1 with keep-alive (so a client's connection pool is
actually reused between calls) and answers two routes:

* ``POST .../messages`` with an Anthropic Messages response
* ``POST .../chat/completions`` with an OpenAI chat completion

It only ever binds the loopback interface and never talks to a real provider.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

GOOD_REPLY: dict[str, str] = {
    "intent": "request_help",
    "response_text": "Of course, what do you need?",
    "suggested_emotion": "empathetic",
}


@dataclass
class RecordedRequest:
    """One request the fake server received."""

    method: str
    path: str
    headers: dict[str, str]
    body: dict[str, Any]


@dataclass
class FakeAPIServer:
    """Loopback server whose next replies are set by the test."""

    requests: list[RecordedRequest] = field(default_factory=list)
    anthropic_content: list[dict[str, Any]] = field(default_factory=list)
    anthropic_stop_reason: str = "end_turn"
    openai_message: dict[str, Any] = field(default_factory=dict)
    _server: ThreadingHTTPServer | None = None
    _thread: threading.Thread | None = None

    def __post_init__(self) -> None:
        self.reply_text(json.dumps(GOOD_REPLY))

    # -- what the server answers ------------------------------------------

    def reply_text(self, text: str) -> None:
        """Answer both routes with ``text`` as the model's reply."""
        self.anthropic_content = [{"type": "text", "text": text}]
        self.anthropic_stop_reason = "end_turn"
        self.openai_message = {"role": "assistant", "content": text}

    def _anthropic_body(self) -> dict[str, Any]:
        return {
            "id": "msg_fake",
            "type": "message",
            "role": "assistant",
            "model": "fake-model",
            "content": self.anthropic_content,
            "stop_reason": self.anthropic_stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    def _openai_body(self) -> dict[str, Any]:
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 1,
            "model": "fake-model",
            "choices": [
                {"index": 0, "message": self.openai_message, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def reply_for(self, path: str) -> dict[str, Any]:
        if path.rstrip("/").endswith("/messages"):
            return self._anthropic_body()
        return self._openai_body()

    # -- lifecycle ---------------------------------------------------------

    @property
    def url(self) -> str:
        assert self._server is not None, "server not started"
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> FakeAPIServer:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                fake.requests.append(
                    RecordedRequest(
                        method=self.command,
                        path=self.path,
                        headers={k.lower(): v for k, v in self.headers.items()},
                        body=json.loads(raw),
                    )
                )
                payload = json.dumps(fake.reply_for(self.path)).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            block_on_close = False

        self._server = Server(("127.0.0.1", 0), Handler)
        # A short poll interval keeps shutdown() (called after every test) quick.
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
