"""How every adapter turns a model reply into an ``InterpretationResult``.

The same reply matrix runs against all four call paths (Claude, OpenAI, a local
OpenAI-compatible server, llama.cpp) with the SDKs faked out, so it runs in CI
without any provider package.  A reply either becomes a result or an
``LLMError``; it must never surface as ``KeyError``, ``AttributeError``,
``IndexError`` or a value of the wrong type.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from intent_engine.errors import LLMError
from intent_engine.llm.base import InterpretationResult
from intent_engine.llm.claude import ClaudeLLM
from intent_engine.llm.local import LocalLLM
from intent_engine.llm.openai import OpenAILLM
from intent_engine.llm.prompts import PROMPT_VERSION
from tests.llm.sdk_mocks import (
    install_anthropic,
    install_openai,
    text_block,
    thinking_block,
    tool_use_block,
)

IML = "<utterance>hello</utterance>"
GOOD = {"intent": "request_help", "response_text": "Sure.", "suggested_emotion": "empathetic"}
GOOD_JSON = json.dumps(GOOD)

Driver = Callable[[pytest.MonkeyPatch, str], InterpretationResult]


def _claude(mp: pytest.MonkeyPatch, reply: str) -> InterpretationResult:
    install_anthropic(mp, [text_block(reply)])
    return asyncio.run(ClaudeLLM(api_key="k").interpret(IML))


def _openai(mp: pytest.MonkeyPatch, reply: str) -> InterpretationResult:
    install_openai(mp, reply)
    return asyncio.run(OpenAILLM(api_key="k").interpret(IML))


def _local_server(mp: pytest.MonkeyPatch, reply: str) -> InterpretationResult:
    install_openai(mp, reply)
    return asyncio.run(LocalLLM(base_url="http://localhost:1/v1").interpret(IML))


def _local_llama(mp: pytest.MonkeyPatch, reply: str) -> InterpretationResult:
    llm = LocalLLM(model_path="/m.gguf")
    llm._llama = MagicMock()
    llm._llama.create_chat_completion.return_value = {
        "choices": [{"message": {"content": reply}}]
    }
    return asyncio.run(llm.interpret(IML))


DRIVERS: dict[str, Driver] = {
    "claude": _claude,
    "openai": _openai,
    "local-server": _local_server,
    "local-llama": _local_llama,
}


@pytest.fixture(params=list(DRIVERS))
def run(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Callable[[str], InterpretationResult]:
    driver = DRIVERS[request.param]
    return lambda reply: driver(monkeypatch, reply)


ACCEPTED = {
    "plain": GOOD_JSON,
    "fenced": f"```json\n{GOOD_JSON}\n```",
    "fenced without language": f"```\n{GOOD_JSON}\n```",
    "preamble": f"Here is my analysis:\n{GOOD_JSON}",
    "trailing prose": f"{GOOD_JSON}\nHope that helps!",
    "padded": f"\n\n   {GOOD_JSON}   \n",
    "preamble with braces": f"Note {{this}} first.\n{GOOD_JSON}",
}


@pytest.mark.parametrize("reply", list(ACCEPTED.values()), ids=list(ACCEPTED))
def test_tolerated_reply_shapes(run: Callable[[str], InterpretationResult], reply: str) -> None:
    assert run(reply) == InterpretationResult(**GOOD)


def _without(key: str) -> str:
    return json.dumps({k: v for k, v in GOOD.items() if k != key})


REJECTED = {
    "empty": "",
    "blank": "   \n",
    "prose": "I cannot help with that.",
    "truncated": '{"intent": "request_he',
    "list": "[1, 2]",
    "list of objects": f"[{GOOD_JSON}]",
    "string": '"hello"',
    "null": "null",
    "number": "42",
    "missing intent": _without("intent"),
    "missing response_text": _without("response_text"),
    "missing suggested_emotion": _without("suggested_emotion"),
    "null intent": json.dumps({**GOOD, "intent": None}),
    "null response_text": json.dumps({**GOOD, "response_text": None}),
    "null suggested_emotion": json.dumps({**GOOD, "suggested_emotion": None}),
    "numeric response_text": json.dumps({**GOOD, "response_text": 42}),
    "list emotion": json.dumps({**GOOD, "suggested_emotion": ["calm"]}),
    "dict intent": json.dumps({**GOOD, "intent": {"a": 1}}),
    "empty response_text": json.dumps({**GOOD, "response_text": "  "}),
    "empty intent": json.dumps({**GOOD, "intent": ""}),
}


@pytest.mark.parametrize("reply", list(REJECTED.values()), ids=list(REJECTED))
def test_bad_replies_are_llm_errors(
    run: Callable[[str], InterpretationResult], reply: str
) -> None:
    with pytest.raises(LLMError):
        run(reply)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Sad", "sad"),
        ("  CALM ", "calm"),
        ("empathetic", "empathetic"),
        ("excited", "neutral"),
        ("very sad indeed", "neutral"),
    ],
)
def test_suggested_emotion_is_normalised(
    run: Callable[[str], InterpretationResult], raw: str, expected: str
) -> None:
    result = run(json.dumps({**GOOD, "suggested_emotion": raw}))
    assert result.suggested_emotion == expected


def test_success_is_logged_without_intent_or_emotion(
    run: Callable[[str], InterpretationResult], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="intent_engine.llm"):
        run(GOOD_JSON)

    assert PROMPT_VERSION in caplog.text
    assert GOOD["suggested_emotion"] not in caplog.text
    assert GOOD["intent"] not in caplog.text


class TestClaudeContentBlocks:
    def _run(self, mp: pytest.MonkeyPatch, blocks: list[object]) -> InterpretationResult:
        install_anthropic(mp, blocks)
        return asyncio.run(ClaudeLLM(api_key="k").interpret(IML))

    def test_thinking_block_before_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._run(monkeypatch, [thinking_block(), text_block(GOOD_JSON)])
        assert result == InterpretationResult(**GOOD)

    def test_text_split_over_several_blocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        half = len(GOOD_JSON) // 2
        blocks = [text_block(GOOD_JSON[:half]), text_block(GOOD_JSON[half:])]
        assert self._run(monkeypatch, blocks) == InterpretationResult(**GOOD)

    @pytest.mark.parametrize(
        "blocks",
        [[], [thinking_block()], [tool_use_block()], [text_block("")]],
        ids=["empty", "thinking only", "tool use only", "empty text"],
    )
    def test_no_text_is_an_llm_error(
        self, monkeypatch: pytest.MonkeyPatch, blocks: list[object]
    ) -> None:
        with pytest.raises(LLMError, match="no text"):
            self._run(monkeypatch, blocks)

    def test_refusal_is_an_llm_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_anthropic(monkeypatch, [], stop_reason="refusal")
        with pytest.raises(LLMError, match="refus"):
            asyncio.run(ClaudeLLM(api_key="k").interpret(IML))

    def test_truncation_names_max_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_anthropic(monkeypatch, [text_block('{"intent": "req')], stop_reason="max_tokens")
        with pytest.raises(LLMError, match="max_tokens"):
            asyncio.run(ClaudeLLM(api_key="k", max_tokens=64).interpret(IML))


class TestOpenAIStyleCompletions:
    @pytest.mark.parametrize("provider", ["openai", "local-server"])
    def test_no_choices_is_an_llm_error(
        self, monkeypatch: pytest.MonkeyPatch, provider: str
    ) -> None:
        install_openai(monkeypatch, None, choices=[])
        llm = (
            OpenAILLM(api_key="k")
            if provider == "openai"
            else LocalLLM(base_url="http://localhost:1/v1")
        )
        with pytest.raises(LLMError, match="no choices"):
            asyncio.run(llm.interpret(IML))

    @pytest.mark.parametrize("provider", ["openai", "local-server"])
    def test_refusal_message_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, provider: str
    ) -> None:
        install_openai(monkeypatch, None, refusal="I can't help with that.")
        llm = (
            OpenAILLM(api_key="k")
            if provider == "openai"
            else LocalLLM(base_url="http://localhost:1/v1")
        )
        with pytest.raises(LLMError, match="can't help"):
            asyncio.run(llm.interpret(IML))

    def test_llama_without_choices_is_an_llm_error(self) -> None:
        llm = LocalLLM(model_path="/m.gguf")
        llm._llama = MagicMock()
        llm._llama.create_chat_completion.return_value = {"choices": []}
        with pytest.raises(LLMError, match="no choices"):
            asyncio.run(llm.interpret(IML))
