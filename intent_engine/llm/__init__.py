"""LLM provider adapters.

Provides a provider-agnostic interface for intent interpretation
from IML-annotated text. All adapters consume IML strings from
``prosody_protocol.IMLParser`` and return ``InterpretationResult``
objects.

Usage::

    from intent_engine.llm import create_llm_provider

    llm = create_llm_provider("claude", api_key="sk-...")
    result = await llm.interpret(iml_string)
"""

from __future__ import annotations

from typing import Any

from intent_engine.llm.base import InterpretationResult, LLMProvider

__all__ = [
    "LLMProvider",
    "InterpretationResult",
    "LLM_PROVIDERS",
    "create_llm_provider",
]


def _get_provider_class(name: str) -> type[LLMProvider]:
    """Lazily import provider classes to avoid requiring all SDKs at once."""
    if name == "claude":
        from intent_engine.llm.claude import ClaudeLLM

        return ClaudeLLM
    if name == "openai":
        from intent_engine.llm.openai import OpenAILLM

        return OpenAILLM
    if name == "local":
        from intent_engine.llm.local import LocalLLM

        return LocalLLM
    raise ValueError(
        f"Unknown LLM provider: {name!r}. "
        f"Available providers: {', '.join(LLM_PROVIDERS)}"
    )


LLM_PROVIDERS: dict[str, str] = {
    "claude": "intent_engine.llm.claude.ClaudeLLM",
    "openai": "intent_engine.llm.openai.OpenAILLM",
    "local": "intent_engine.llm.local.LocalLLM",
}
"""Registry of available LLM provider names."""


def create_llm_provider(name: str, **kwargs: Any) -> LLMProvider:
    """Create an LLM provider instance by name.

    Parameters
    ----------
    name:
        Provider name.  One of ``"claude"``, ``"openai"``, ``"local"``.
    **kwargs:
        Provider-specific configuration passed to the constructor.

    Returns
    -------
    LLMProvider
        An initialized LLM provider instance.
    """
    cls = _get_provider_class(name)
    return cls(**kwargs)
