"""Anthropic Claude LLM adapter.

Uses the ``anthropic`` Python SDK to send IML-annotated messages
to Claude and parse structured JSON responses for intent interpretation.
Requires an ``ANTHROPIC_API_KEY`` environment variable or explicit
``api_key`` parameter.
"""

from __future__ import annotations

import json
import logging
import os

from intent_engine.llm.base import InterpretationResult, LLMProvider
from intent_engine.llm.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ClaudeLLM(LLMProvider):
    """Anthropic Claude-based LLM provider.

    Parameters
    ----------
    api_key:
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY``
        environment variable if not provided.
    model:
        Claude model to use (e.g., ``"claude-sonnet-4-20250514"``).
    max_tokens:
        Maximum tokens in the response.
    temperature:
        Sampling temperature (0.0-1.0).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: object,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY or pass api_key=."
            )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def interpret(
        self, iml_input: str, context: str | None = None
    ) -> InterpretationResult:
        """Interpret IML-annotated input using Claude.

        Parameters
        ----------
        iml_input:
            Serialized IML markup string.
        context:
            Optional conversation context appended to the system prompt.

        Returns
        -------
        InterpretationResult
            The parsed intent, response text, and suggested emotion.
        """
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "anthropic is required for ClaudeLLM. "
                "Install it with: pip install intent-engine[claude]"
            ) from exc

        system = SYSTEM_PROMPT
        if context:
            system = f"{system}\n\n## Additional Context\n{context}"

        client = anthropic.AsyncAnthropic(api_key=self._api_key)

        response = await client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system,
            messages=[{"role": "user", "content": iml_input}],
        )

        raw_text = response.content[0].text
        parsed = json.loads(raw_text)

        logger.info(
            "Claude interpreted intent=%s emotion=%s (model=%s)",
            parsed.get("intent"),
            parsed.get("suggested_emotion"),
            self._model,
        )

        return InterpretationResult(
            intent=parsed["intent"],
            response_text=parsed["response_text"],
            suggested_emotion=parsed["suggested_emotion"],
        )
