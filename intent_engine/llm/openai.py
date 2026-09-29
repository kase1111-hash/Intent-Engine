"""OpenAI LLM adapter.

Uses the ``openai`` Python SDK to send IML-annotated messages
to GPT models and parse structured JSON responses for intent
interpretation. Requires an ``OPENAI_API_KEY`` environment variable
or explicit ``api_key`` parameter.
"""

from __future__ import annotations

import logging
import os

from intent_engine.llm.base import (
    InterpretationResult,
    LLMProvider,
    chat_completion_text,
    parse_interpretation,
)
from intent_engine.llm.prompts import PROMPT_VERSION, SYSTEM_PROMPT

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

        Raises
        ------
        intent_engine.errors.LLMError
            If the model refuses or does not reply with the expected JSON.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAILLM. "
                "Install it with: pip install intent-engine[openai]"
            ) from exc

        system = SYSTEM_PROMPT
        if context:
            system = f"{system}\n\n## Additional Context\n{context}"

        # The client is opened per call: the sync wrappers run every call in a
        # fresh event loop (asyncio.run), and a client keeps its connections
        # bound to the loop that first used it.
        async with AsyncOpenAI(api_key=self._api_key) as client:
            response = await client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": iml_input},
                ],
            )

        result = parse_interpretation(chat_completion_text(response, "OpenAI"), "OpenAI")

        logger.info(
            "OpenAI interpreted a turn (model=%s, prompt=%s)", self._model, PROMPT_VERSION
        )
        return result
