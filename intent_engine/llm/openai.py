"""OpenAI LLM adapter.

Uses the ``openai`` Python SDK to send IML-annotated messages
to GPT models and parse structured JSON responses for intent
interpretation. Requires an ``OPENAI_API_KEY`` environment variable
or explicit ``api_key`` parameter.
"""

from __future__ import annotations

import json
import logging
import os

from intent_engine.llm.base import InterpretationResult, LLMProvider
from intent_engine.llm.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class OpenAILLM(LLMProvider):
    """OpenAI GPT-based LLM provider.

    Parameters
    ----------
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY``
        environment variable if not provided.
    model:
        OpenAI model to use (e.g., ``"gpt-4o"``).
    max_tokens:
        Maximum tokens in the response.
    temperature:
        Sampling temperature (0.0-2.0).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: object,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY or pass api_key=."
            )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client: object | None = None

    def _get_client(self) -> object:
        """Return a reusable async client, creating it on first call."""
        if self._client is None:
            from openai import AsyncOpenAI  # type: ignore[import-untyped]

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def interpret(
        self, iml_input: str, context: str | None = None
    ) -> InterpretationResult:
        """Interpret IML-annotated input using an OpenAI model.

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
            client = self._get_client()
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAILLM. "
                "Install it with: pip install intent-engine[openai]"
            ) from exc

        system = SYSTEM_PROMPT
        if context:
            system = f"{system}\n\n## Additional Context\n{context}"

        response = await client.chat.completions.create(  # type: ignore[union-attr]
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": iml_input},
            ],
        )

        raw_text = response.choices[0].message.content or ""

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            from intent_engine.errors import LLMError

            raise LLMError(
                f"OpenAI returned non-JSON response: {raw_text[:200]}"
            ) from exc

        logger.info(
            "OpenAI interpreted intent=%s emotion=%s (model=%s)",
            parsed.get("intent"),
            parsed.get("suggested_emotion"),
            self._model,
        )

        return InterpretationResult(
            intent=parsed["intent"],
            response_text=parsed["response_text"],
            suggested_emotion=parsed["suggested_emotion"],
        )
