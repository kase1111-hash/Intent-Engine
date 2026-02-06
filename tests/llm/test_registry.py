"""Tests for the LLM provider registry and factory."""

from __future__ import annotations

import os

import pytest

from intent_engine.llm import LLM_PROVIDERS, LLMProvider, create_llm_provider


class TestLLMProviderRegistry:
    def test_registry_has_expected_providers(self) -> None:
        assert "claude" in LLM_PROVIDERS
        assert "openai" in LLM_PROVIDERS
        assert "local" in LLM_PROVIDERS

    def test_registry_has_three_providers(self) -> None:
        assert len(LLM_PROVIDERS) == 3

    def test_registry_values_are_module_paths(self) -> None:
        for name, path in LLM_PROVIDERS.items():
            assert path.startswith("intent_engine.llm."), (
                f"Provider {name} path should start with intent_engine.llm."
            )


class TestCreateLLMProvider:
    def test_create_claude_provider(self) -> None:
        llm = create_llm_provider("claude", api_key="test-key")
        assert isinstance(llm, LLMProvider)
        from intent_engine.llm.claude import ClaudeLLM

        assert isinstance(llm, ClaudeLLM)

    def test_create_openai_provider(self) -> None:
        llm = create_llm_provider("openai", api_key="test-key")
        assert isinstance(llm, LLMProvider)
        from intent_engine.llm.openai import OpenAILLM

        assert isinstance(llm, OpenAILLM)

    def test_create_local_provider(self) -> None:
        llm = create_llm_provider("local", model_path="/path/to/model.gguf")
        assert isinstance(llm, LLMProvider)
        from intent_engine.llm.local import LocalLLM

        assert isinstance(llm, LocalLLM)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_provider("nonexistent")

    def test_error_message_lists_available_providers(self) -> None:
        with pytest.raises(ValueError, match="claude.*openai.*local"):
            create_llm_provider("bad-name")

    def test_kwargs_passed_to_constructor(self) -> None:
        llm = create_llm_provider(
            "claude",
            api_key="my-key",
            model="claude-opus-4-20250514",
            max_tokens=4096,
        )
        from intent_engine.llm.claude import ClaudeLLM

        assert isinstance(llm, ClaudeLLM)
        assert llm._api_key == "my-key"
        assert llm._model == "claude-opus-4-20250514"
        assert llm._max_tokens == 4096
