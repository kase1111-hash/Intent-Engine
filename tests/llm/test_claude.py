"""Tests for the Claude LLM adapter."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intent_engine.llm.base import InterpretationResult, LLMProvider
from intent_engine.llm.claude import ClaudeLLM


class TestClaudeLLMConstruction:
    def test_requires_api_key(self) -> None:
        old_val = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="Anthropic API key is required"):
                ClaudeLLM()
        finally:
            if old_val is not None:
                os.environ["ANTHROPIC_API_KEY"] = old_val

    def test_accepts_explicit_api_key(self) -> None:
        llm = ClaudeLLM(api_key="test-key-123")
        assert llm._api_key == "test-key-123"

    def test_reads_env_var(self) -> None:
        os.environ["ANTHROPIC_API_KEY"] = "env-key-456"
        try:
            llm = ClaudeLLM()
            assert llm._api_key == "env-key-456"
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    def test_default_model(self) -> None:
        llm = ClaudeLLM(api_key="key")
        assert llm._model == "claude-sonnet-4-20250514"

    def test_custom_params(self) -> None:
        llm = ClaudeLLM(
            api_key="key",
            model="claude-opus-4-20250514",
            max_tokens=2048,
            temperature=0.7,
        )
        assert llm._model == "claude-opus-4-20250514"
        assert llm._max_tokens == 2048
        assert llm._temperature == 0.7

    def test_is_llm_provider(self) -> None:
        llm = ClaudeLLM(api_key="key")
        assert isinstance(llm, LLMProvider)

    def test_accepts_kwargs(self) -> None:
        llm = ClaudeLLM(api_key="key", extra_param="ignored")
        assert llm._api_key == "key"


class TestClaudeLLMInterpret:
    def test_import_error_without_sdk(self) -> None:
        llm = ClaudeLLM(api_key="key")
        with patch.dict(sys.modules, {"anthropic": None}), pytest.raises(
            ImportError, match="anthropic is required"
        ):
            asyncio.run(
                llm.interpret("<utterance>hello</utterance>")
            )

    def test_interpret_returns_interpretation_result(self) -> None:
        mock_anthropic = types.ModuleType("anthropic")

        response_json = json.dumps({
            "intent": "greet",
            "response_text": "Hello! How can I help you?",
            "suggested_emotion": "joyful",
        })

        mock_content_block = MagicMock()
        mock_content_block.text = response_json

        mock_response = MagicMock()
        mock_response.content = [mock_content_block]

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        mock_anthropic.AsyncAnthropic = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
        sys.modules["anthropic"] = mock_anthropic

        try:
            llm = ClaudeLLM(api_key="test-key")
            result = asyncio.run(
                llm.interpret('<utterance emotion="joyful" confidence="0.9">Hello!</utterance>')
            )

            assert isinstance(result, InterpretationResult)
            assert result.intent == "greet"
            assert result.response_text == "Hello! How can I help you?"
            assert result.suggested_emotion == "joyful"
        finally:
            sys.modules.pop("anthropic", None)

    def test_interpret_with_context(self) -> None:
        mock_anthropic = types.ModuleType("anthropic")

        response_json = json.dumps({
            "intent": "ask_question",
            "response_text": "Sure, let me help.",
            "suggested_emotion": "calm",
        })

        mock_content_block = MagicMock()
        mock_content_block.text = response_json

        mock_response = MagicMock()
        mock_response.content = [mock_content_block]

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        mock_anthropic.AsyncAnthropic = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
        sys.modules["anthropic"] = mock_anthropic

        try:
            llm = ClaudeLLM(api_key="test-key")
            result = asyncio.run(
                llm.interpret(
                    "<utterance>Can you help?</utterance>",
                    context="Customer support scenario",
                )
            )

            assert result.intent == "ask_question"

            # Verify context was passed in system prompt
            call_kwargs = mock_client.messages.create.call_args
            system_arg = call_kwargs.kwargs.get("system", "")
            assert "Customer support scenario" in system_arg
        finally:
            sys.modules.pop("anthropic", None)
