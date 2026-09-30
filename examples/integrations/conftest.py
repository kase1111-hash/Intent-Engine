"""Shared fixtures for the integration example tests.

The results and audio built here are the real dataclasses and bytes the
engine produces, not mocks, so the tests exercise the same shapes the
examples see in production.
"""

from __future__ import annotations

import io
import wave
from collections.abc import Callable
from typing import Any

import pytest
from prosody_protocol import IMLParser

from intent_engine.models.result import Result

ResultFactory = Callable[..., Result]
Handler = Callable[[Any], Any]


@pytest.fixture()
def make_result() -> ResultFactory:
    """Build a real ``Result``; ``confidence=0.0`` means no emotion reported."""

    def factory(
        text: str = "Hello",
        emotion: str = "neutral",
        confidence: float = 0.0,
        suggested_tone: str | None = None,
    ) -> Result:
        iml = "<utterance>Hello</utterance>"
        if suggested_tone is None:
            suggested_tone = emotion if confidence >= 0.5 else "neutral"
        return Result(
            text=text,
            emotion=emotion,
            confidence=confidence,
            iml=iml,
            iml_document=IMLParser().parse(iml),
            suggested_tone=suggested_tone,
            prosody_features=[],
        )

    return factory


@pytest.fixture()
def wav_bytes() -> bytes:
    """A valid, silent 0.1 s mono 16 kHz WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


@pytest.fixture()
def mock_httpx(monkeypatch: pytest.MonkeyPatch) -> Callable[[Handler], list[Any]]:
    """Route every ``httpx.AsyncClient`` through a handler; returns the request log.

    No socket is ever opened, so tests can exercise the real download code
    against any URL.
    """
    httpx = pytest.importorskip("httpx")

    def install(handler: Handler) -> list[Any]:
        seen: list[Any] = []
        real = httpx.AsyncClient

        def logging_handler(request: Any) -> Any:
            seen.append(request)
            return handler(request)

        def factory(*args: Any, **kwargs: Any) -> Any:
            return real(*args, transport=httpx.MockTransport(logging_handler), **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", factory)
        return seen

    return install
