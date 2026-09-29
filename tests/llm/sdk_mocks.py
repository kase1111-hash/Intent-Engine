"""Stand-ins for the provider SDKs, for tests that run without them installed.

CI installs only the ``dev`` extras, so the adapters' own logic (reply parsing,
error handling) has to be testable against fakes.  The fake clients support
``async with`` because the adapters open a fresh client for every call.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


def _async_client() -> MagicMock:
    client = MagicMock()
    client.__aenter__.return_value = client
    return client


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def thinking_block() -> SimpleNamespace:
    """A block like the SDK's ``ThinkingBlock``: it has no ``text`` attribute."""
    return SimpleNamespace(type="thinking", thinking="", signature="sig")


def tool_use_block() -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id="toolu_1", name="t", input={})


def install_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    blocks: list[Any],
    stop_reason: str = "end_turn",
) -> MagicMock:
    """Replace ``anthropic`` with a fake whose client replies with ``blocks``."""
    client = _async_client()
    client.messages.create = AsyncMock(
        return_value=SimpleNamespace(content=blocks, stop_reason=stop_reason)
    )
    module = types.ModuleType("anthropic")
    module.AsyncAnthropic = MagicMock(return_value=client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return client


def install_openai(
    monkeypatch: pytest.MonkeyPatch,
    content: str | None,
    refusal: str | None = None,
    choices: list[Any] | None = None,
) -> MagicMock:
    """Replace ``openai`` with a fake whose client replies with ``content``."""
    message = SimpleNamespace(content=content, refusal=refusal)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)] if choices is None else choices
    )
    client = _async_client()
    client.chat.completions.create = AsyncMock(return_value=response)
    module = types.ModuleType("openai")
    module.AsyncOpenAI = MagicMock(return_value=client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    return client
