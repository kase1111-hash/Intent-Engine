"""Fixtures for the LLM adapter tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.llm.fake_server import FakeAPIServer


@pytest.fixture()
def fake_api(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeAPIServer]:
    """A loopback fake provider API that the SDKs reach via their base-URL env vars."""
    # The SDKs honour proxy variables; keep loopback traffic off any proxy.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    server = FakeAPIServer().start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server.url)
    monkeypatch.setenv("OPENAI_BASE_URL", f"{server.url}/v1")
    try:
        yield server
    finally:
        server.stop()
