"""Tests for the OpenAI LLM adapter."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intent_engine.llm.base import InterpretationResult, LLMProvider
from intent_engine.llm.openai import OpenAILLM


class TestOpenAILLMConstruction:
    def test_requires_api_key(self) -> None:
        old_val = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                OpenAILLM()
        finally:
            if old_val is not None:
                os.environ["OPENAI_API_KEY"] = old_val

    def test_accepts_explicit_api_key(self) -> None:
        llm = OpenAILLM(api_key="test-key-123")
        assert llm._api_key == "test-key-123"

    def test_reads_env_var(self) -> None:
        os.environ["OPENAI_API_KEY"] = "env-key-456"
        try:
            llm = OpenAILLM()
            assert llm._api_key == "env-key-456"
        finally:
            del os.environ["OPENAI_API_KEY"]

    def test_default_model(self) -> None:
        llm = OpenAILLM(api_key="key")
        assert llm._model == "gpt-4o"

    def test_custom_params(self) -> None:
        llm = OpenAILLM(
            api_key="key",
            model="gpt-4-turbo",
            max_tokens=2048,
            temperature=0.5,
        )
        assert llm._model == "gpt-4-turbo"
        assert llm._max_tokens == 2048
        assert llm._temperature == 0.5

    def test_is_llm_provider(self) -> None:
        llm = OpenAILLM(api_key="key")
        assert isinstance(llm, LLMProvider)

    def test_accepts_kwargs(self) -> None:
        llm = OpenAILLM(api_key="key", extra_param="ignored")
        assert llm._api_key == "key"


class TestOpenAILLMInterpret:
    def test_import_error_without_sdk(self) -> None:
        llm = OpenAILLM(api_key="key")
        with patch.dict(sys.modules, {"openai": None}), pytest.raises(
            ImportError, match="openai is required"
        ):
            asyncio.run(
                llm.interpret("<utterance>hello</utterance>")
            )

    def test_interpret_returns_interpretation_result(self) -> None:
        mock_openai = types.ModuleType("openai")

        response_json = json.dumps({
            "intent": "express_frustration",
            "response_text": "I understand your frustration.",
            "suggested_emotion": "empathetic",
        })

        mock_message = MagicMock()
        mock_message.content = response_json

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_completions = MagicMock()
        mock_completions.create = AsyncMock(return_value=mock_response)

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_client = MagicMock()
        mock_client.chat = mock_chat

        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
        sys.modules["openai"] = mock_openai

        try:
            llm = OpenAILLM(api_key="test-key")
            result = asyncio.run(
                llm.interpret(
                    '<utterance emotion="frustrated" confidence="0.8">'
                    "This is really annoying!</utterance>"
                )
            )

            assert isinstance(result, InterpretationResult)
            assert result.intent == "express_frustration"
            assert result.response_text == "I understand your frustration."
            assert result.suggested_emotion == "empathetic"
        finally:
            sys.modules.pop("openai", None)

    def test_interpret_uses_json_mode(self) -> None:
        mock_openai = types.ModuleType("openai")

        response_json = json.dumps({
            "intent": "test",
            "response_text": "test",
            "suggested_emotion": "neutral",
        })

        mock_message = MagicMock()
        mock_message.content = response_json

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_completions = MagicMock()
        mock_completions.create = AsyncMock(return_value=mock_response)

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_client = MagicMock()
        mock_client.chat = mock_chat

        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
        sys.modules["openai"] = mock_openai

        try:
            llm = OpenAILLM(api_key="test-key")
            asyncio.run(
                llm.interpret("<utterance>hello</utterance>")
            )

            call_kwargs = mock_completions.create.call_args.kwargs
            assert call_kwargs["response_format"] == {"type": "json_object"}
        finally:
            sys.modules.pop("openai", None)

    def test_interpret_with_context(self) -> None:
        mock_openai = types.ModuleType("openai")

        response_json = json.dumps({
            "intent": "test",
            "response_text": "test",
            "suggested_emotion": "neutral",
        })

        mock_message = MagicMock()
        mock_message.content = response_json

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_completions = MagicMock()
        mock_completions.create = AsyncMock(return_value=mock_response)

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_client = MagicMock()
        mock_client.chat = mock_chat

        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
        sys.modules["openai"] = mock_openai

        try:
            llm = OpenAILLM(api_key="test-key")
            asyncio.run(
                llm.interpret(
                    "<utterance>hello</utterance>",
                    context="Medical triage scenario",
                )
            )

            call_kwargs = mock_completions.create.call_args.kwargs
            messages = call_kwargs["messages"]
            system_content = messages[0]["content"]
            assert "Medical triage scenario" in system_content
        finally:
            sys.modules.pop("openai", None)
