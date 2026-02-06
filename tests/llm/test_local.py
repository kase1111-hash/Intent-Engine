"""Tests for the Local LLM adapter."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intent_engine.llm.base import InterpretationResult, LLMProvider
from intent_engine.llm.local import LocalLLM


class TestLocalLLMConstruction:
    def test_requires_model_path_or_base_url(self) -> None:
        with pytest.raises(ValueError, match="Either model_path .* or base_url"):
            LocalLLM()

    def test_accepts_model_path(self) -> None:
        llm = LocalLLM(model_path="/path/to/model.gguf")
        assert llm._model_path == "/path/to/model.gguf"
        assert llm._base_url is None

    def test_accepts_base_url(self) -> None:
        llm = LocalLLM(base_url="http://localhost:11434/v1")
        assert llm._base_url == "http://localhost:11434/v1"
        assert llm._model_path is None

    def test_default_params(self) -> None:
        llm = LocalLLM(model_path="/path/to/model.gguf")
        assert llm._model == "llama3"
        assert llm._n_ctx == 4096
        assert llm._n_gpu_layers == 0
        assert llm._max_tokens == 1024
        assert llm._temperature == 0.3

    def test_custom_params(self) -> None:
        llm = LocalLLM(
            model_path="/path/to/model.gguf",
            model="custom",
            n_ctx=8192,
            n_gpu_layers=32,
            max_tokens=2048,
            temperature=0.5,
        )
        assert llm._n_ctx == 8192
        assert llm._n_gpu_layers == 32
        assert llm._max_tokens == 2048
        assert llm._temperature == 0.5

    def test_is_llm_provider(self) -> None:
        llm = LocalLLM(model_path="/path/to/model.gguf")
        assert isinstance(llm, LLMProvider)

    def test_accepts_kwargs(self) -> None:
        llm = LocalLLM(model_path="/m.gguf", extra_param="ignored")
        assert llm._model_path == "/m.gguf"

    def test_lazy_load_not_triggered_on_init(self) -> None:
        llm = LocalLLM(model_path="/path/to/model.gguf")
        assert llm._llama is None


class TestLocalLLMLlamaBackend:
    def test_import_error_without_llama_cpp(self) -> None:
        llm = LocalLLM(model_path="/path/to/model.gguf")
        with patch.dict(sys.modules, {"llama_cpp": None}), pytest.raises(
            ImportError, match="llama-cpp-python is required"
        ):
            llm._load_llama()

    def test_interpret_via_llama(self) -> None:
        mock_llama_cpp = types.ModuleType("llama_cpp")

        response_json = json.dumps({
            "intent": "request_info",
            "response_text": "Here is the information.",
            "suggested_emotion": "neutral",
        })

        mock_llama_instance = MagicMock()
        mock_llama_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": response_json}}],
        }

        mock_llama_cpp.Llama = MagicMock(return_value=mock_llama_instance)  # type: ignore[attr-defined]
        sys.modules["llama_cpp"] = mock_llama_cpp

        try:
            llm = LocalLLM(model_path="/path/to/model.gguf")
            # Pre-load to avoid file access
            llm._llama = mock_llama_instance

            result = asyncio.get_event_loop().run_until_complete(
                llm.interpret("<utterance>What is the weather?</utterance>")
            )

            assert isinstance(result, InterpretationResult)
            assert result.intent == "request_info"
            assert result.response_text == "Here is the information."
            assert result.suggested_emotion == "neutral"
        finally:
            sys.modules.pop("llama_cpp", None)

    def test_llama_uses_json_mode(self) -> None:
        response_json = json.dumps({
            "intent": "test",
            "response_text": "test",
            "suggested_emotion": "neutral",
        })

        mock_llama_instance = MagicMock()
        mock_llama_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": response_json}}],
        }

        llm = LocalLLM(model_path="/path/to/model.gguf")
        llm._llama = mock_llama_instance

        asyncio.get_event_loop().run_until_complete(
            llm.interpret("<utterance>hello</utterance>")
        )

        call_kwargs = mock_llama_instance.create_chat_completion.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}


class TestLocalLLMServerBackend:
    def test_import_error_without_openai(self) -> None:
        llm = LocalLLM(base_url="http://localhost:11434/v1")
        with patch.dict(sys.modules, {"openai": None}), pytest.raises(
            ImportError, match="openai is required"
        ):
            asyncio.get_event_loop().run_until_complete(
                llm.interpret("<utterance>hello</utterance>")
            )

    def test_interpret_via_server(self) -> None:
        mock_openai = types.ModuleType("openai")

        response_json = json.dumps({
            "intent": "confirm_order",
            "response_text": "Order confirmed.",
            "suggested_emotion": "calm",
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
            llm = LocalLLM(base_url="http://localhost:11434/v1", model="mistral")
            result = asyncio.get_event_loop().run_until_complete(
                llm.interpret("<utterance>Confirm my order</utterance>")
            )

            assert isinstance(result, InterpretationResult)
            assert result.intent == "confirm_order"
            assert result.suggested_emotion == "calm"

            # Verify the client was created with correct base_url
            mock_openai.AsyncOpenAI.assert_called_once_with(  # type: ignore[attr-defined]
                api_key="not-needed",
                base_url="http://localhost:11434/v1",
            )
        finally:
            sys.modules.pop("openai", None)

    def test_server_uses_correct_model(self) -> None:
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
            llm = LocalLLM(base_url="http://localhost:8000/v1", model="qwen2")
            asyncio.get_event_loop().run_until_complete(
                llm.interpret("<utterance>hello</utterance>")
            )

            call_kwargs = mock_completions.create.call_args.kwargs
            assert call_kwargs["model"] == "qwen2"
        finally:
            sys.modules.pop("openai", None)


class TestLocalLLMWithContext:
    def test_context_appended_to_system_prompt(self) -> None:
        response_json = json.dumps({
            "intent": "test",
            "response_text": "test",
            "suggested_emotion": "neutral",
        })

        mock_llama_instance = MagicMock()
        mock_llama_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": response_json}}],
        }

        llm = LocalLLM(model_path="/path/to/model.gguf")
        llm._llama = mock_llama_instance

        asyncio.get_event_loop().run_until_complete(
            llm.interpret(
                "<utterance>hello</utterance>",
                context="Healthcare triage system",
            )
        )

        call_kwargs = mock_llama_instance.create_chat_completion.call_args.kwargs
        messages = call_kwargs["messages"]
        system_content = messages[0]["content"]
        assert "Healthcare triage system" in system_content
