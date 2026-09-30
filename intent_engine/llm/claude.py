"""Anthropic Claude LLM adapter.

Uses the ``anthropic`` Python SDK to send IML-annotated messages
to Claude and parse structured JSON responses for intent interpretation.
Requires an ``ANTHROPIC_API_KEY`` environment variable or explicit
``api_key`` parameter.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from intent_engine.errors import LLMError
from intent_engine.llm.base import InterpretationResult, LLMProvider, parse_interpretation
from intent_engine.llm.prompts import PROMPT_VERSION, SYSTEM_PROMPT

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
        Maximum tokens in the response.  Current Claude models think
        before they answer and the thinking counts towards this limit,
        so a small limit can be used up by the thinking and truncate the
        JSON reply (reported as an ``LLMError``).  The default is the
        most the SDK allows without streaming for every model.
    temperature:
        Optional sampling temperature.  Current ``anthropic`` SDKs
        (>= 1.0) and current Claude models no longer accept sampling
        parameters, so nothing is sent unless this is set.  When it is
        set the value is passed in the request body (``extra_body``),
        which only works with a model that still accepts it.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8192,
        temperature: float | None = None,
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

        Raises
        ------
        intent_engine.errors.LLMError
            If Claude refuses, is cut off, or does not reply with the
            expected JSON.
        """
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic is required for ClaudeLLM. "
                "Install it with: pip install intent-engine[claude]"
            ) from exc

        system = SYSTEM_PROMPT
        if context:
            system = f"{system}\n\n## Additional Context\n{context}"

        extra: dict[str, Any] = {}
        if self._temperature is not None:
            extra["extra_body"] = {"temperature": self._temperature}

        # The client is opened per call, not cached: an async client keeps its
        # connections bound to the event loop that first used it, and this adapter
        # can be awaited from a different loop each time (a caller that runs
        # asyncio.run per call, or an engine whose sync-wrapper loop was
        # recreated after close() or a fork).
        async with anthropic.AsyncAnthropic(api_key=self._api_key) as client:
            response = await client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": iml_input}],
                **extra,
            )

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            raise LLMError("Claude declined to answer (stop_reason=refusal)")

        # Models that think put thinking blocks in front of the text block.
        raw_text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        cut_off = (
            f"Claude's reply was cut off at max_tokens={self._max_tokens} "
            "(thinking counts towards it); raise max_tokens"
        )
        if not raw_text.strip():
            # Thinking that uses up the whole budget leaves no text block at all.
            if stop_reason == "max_tokens":
                raise LLMError(cut_off)
            raise LLMError("Claude returned no text content")

        try:
            result = parse_interpretation(raw_text, "Claude")
        except LLMError as exc:
            if stop_reason == "max_tokens":
                raise LLMError(cut_off) from exc
            raise

        logger.info(
            "Claude interpreted a turn (model=%s, prompt=%s)", self._model, PROMPT_VERSION
        )
        return result
