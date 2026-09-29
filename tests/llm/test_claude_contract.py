"""ClaudeLLM against the real ``anthropic`` SDK and a loopback fake API.

The mocked tests in ``test_claude.py`` accept any keyword arguments, so they
cannot notice when the SDK's ``messages.create`` signature moves on.  These
tests let the installed SDK build and send the request, then check what
arrived.  They are skipped when ``anthropic`` is not installed.
"""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest

anthropic = pytest.importorskip("anthropic")

from intent_engine.errors import LLMError  # noqa: E402
from intent_engine.llm.base import InterpretationResult  # noqa: E402
from intent_engine.llm.claude import ClaudeLLM  # noqa: E402
from tests.llm.fake_server import GOOD_REPLY, FakeAPIServer  # noqa: E402

IML = '<utterance emotion="joyful" confidence="0.9">Hello!</utterance>'


def _interpret(llm: ClaudeLLM, iml: str = IML) -> InterpretationResult:
    return asyncio.run(llm.interpret(iml))


class TestRequestShape:
    def test_interpret_succeeds_with_installed_sdk(self, fake_api: FakeAPIServer) -> None:
        result = _interpret(ClaudeLLM(api_key="k"))

        assert result == InterpretationResult(**GOOD_REPLY)
        assert len(fake_api.requests) == 1

    def test_no_temperature_is_sent_by_default(self, fake_api: FakeAPIServer) -> None:
        _interpret(ClaudeLLM(api_key="k"))

        assert "temperature" not in fake_api.requests[0].body

    def test_only_parameters_the_sdk_declares_are_passed(
        self, fake_api: FakeAPIServer
    ) -> None:
        _interpret(ClaudeLLM(api_key="k"))

        client = anthropic.AsyncAnthropic(api_key="k")
        declared = set(inspect.signature(client.messages.create).parameters)
        sent = set(fake_api.requests[0].body)
        assert sent <= declared | {"stream"}, sent - declared

    def test_the_configured_model_is_what_is_sent(self, fake_api: FakeAPIServer) -> None:
        _interpret(ClaudeLLM(api_key="k", model="the-configured-model"))

        assert fake_api.requests[0].body["model"] == "the-configured-model"

    def test_default_max_tokens_leaves_room_for_thinking(
        self, fake_api: FakeAPIServer
    ) -> None:
        _interpret(ClaudeLLM(api_key="k"))

        assert fake_api.requests[0].body["max_tokens"] >= 4096

    @pytest.mark.filterwarnings("ignore:The model:DeprecationWarning")
    @pytest.mark.parametrize("model", ["claude-opus-4-20250514", "claude-opus-4-0"])
    def test_default_max_tokens_needs_no_streaming_on_any_model(
        self, fake_api: FakeAPIServer, model: str
    ) -> None:
        # The SDK refuses non-streaming requests above 8192 tokens for some models.
        _interpret(ClaudeLLM(api_key="k", model=model))

        assert fake_api.requests[0].body["model"] == model

    def test_explicit_temperature_goes_through_extra_body(
        self, fake_api: FakeAPIServer
    ) -> None:
        _interpret(ClaudeLLM(api_key="k", temperature=0.7))

        assert fake_api.requests[0].body["temperature"] == 0.7

    def test_system_prompt_and_context_are_sent(self, fake_api: FakeAPIServer) -> None:
        asyncio.run(ClaudeLLM(api_key="k").interpret(IML, context="Support desk"))

        body = fake_api.requests[0].body
        assert "Support desk" in body["system"]
        assert body["messages"] == [{"role": "user", "content": IML}]


class TestResponseShapes:
    def test_thinking_block_before_text_is_skipped(self, fake_api: FakeAPIServer) -> None:
        fake_api.anthropic_content = [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": json.dumps(GOOD_REPLY)},
        ]

        assert _interpret(ClaudeLLM(api_key="k")).intent == "request_help"

    def test_empty_content_is_an_llm_error(self, fake_api: FakeAPIServer) -> None:
        fake_api.anthropic_content = []

        with pytest.raises(LLMError, match="no text"):
            _interpret(ClaudeLLM(api_key="k"))

    def test_refusal_is_an_llm_error(self, fake_api: FakeAPIServer) -> None:
        fake_api.anthropic_content = []
        fake_api.anthropic_stop_reason = "refusal"

        with pytest.raises(LLMError, match="refus"):
            _interpret(ClaudeLLM(api_key="k"))

    def test_truncated_reply_says_max_tokens_was_hit(self, fake_api: FakeAPIServer) -> None:
        fake_api.anthropic_content = [{"type": "text", "text": '{"intent": "request_he'}]
        fake_api.anthropic_stop_reason = "max_tokens"

        with pytest.raises(LLMError, match="max_tokens"):
            _interpret(ClaudeLLM(api_key="k", max_tokens=64))
