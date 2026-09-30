"""The adapters must work when every call runs in a new event loop.

A caller can run each call in a fresh loop, as repeated ``asyncio.run(llm.interpret(...))``
does.  (``IntentEngine``'s ``*_sync`` wrappers share one loop and do not exercise this.)
An async HTTP client keeps its pooled connections bound to the loop that first used
them, so an adapter that caches its client fails on the next loop with
``RuntimeError: Event loop is closed``.  The fake server keeps connections
alive so that a pooled connection really is there to be reused.
"""

from __future__ import annotations

import asyncio

import pytest

from intent_engine.llm.base import InterpretationResult, LLMProvider
from intent_engine.llm.claude import ClaudeLLM
from intent_engine.llm.local import LocalLLM
from intent_engine.llm.openai import OpenAILLM
from tests.llm.fake_server import GOOD_REPLY, FakeAPIServer

IML = "<utterance>hello</utterance>"


def _adapter(provider: str, server: FakeAPIServer) -> LLMProvider:
    if provider == "claude":
        pytest.importorskip("anthropic")
        return ClaudeLLM(api_key="k")
    pytest.importorskip("openai")
    if provider == "openai":
        return OpenAILLM(api_key="k")
    return LocalLLM(base_url=f"{server.url}/v1")


@pytest.mark.parametrize("provider", ["claude", "openai", "local-server"])
def test_repeated_asyncio_run_on_one_adapter(fake_api: FakeAPIServer, provider: str) -> None:
    llm = _adapter(provider, fake_api)

    for _ in range(3):
        result = asyncio.run(llm.interpret(IML))
        assert result == InterpretationResult(**GOOD_REPLY)

    assert len(fake_api.requests) == 3


@pytest.mark.parametrize("provider", ["claude", "openai", "local-server"])
def test_concurrent_calls_in_one_loop(fake_api: FakeAPIServer, provider: str) -> None:
    llm = _adapter(provider, fake_api)

    async def run_three() -> list[InterpretationResult]:
        return list(await asyncio.gather(*(llm.interpret(IML) for _ in range(3))))

    results = asyncio.run(run_three())

    assert results == [InterpretationResult(**GOOD_REPLY)] * 3
