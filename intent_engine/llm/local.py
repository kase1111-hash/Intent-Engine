"""Local LLM adapter -- llama.cpp / vLLM / Ollama.

Supports two backends:

1. **llama.cpp** via ``llama-cpp-python``: Load a GGUF model file
   directly and run inference locally.
2. **OpenAI-compatible server** (vLLM, Ollama, etc.): Connect to a
   local HTTP endpoint that exposes an OpenAI-compatible chat API.

No external API key required -- everything runs on the local machine.
"""

from __future__ import annotations

import logging
from typing import Any

from intent_engine.llm.base import (
    InterpretationResult,
    LLMProvider,
    chat_completion_text,
    parse_interpretation,
)
from intent_engine.llm.prompts import PROMPT_VERSION, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LocalLLM(LLMProvider):
    """Local LLM provider using llama.cpp or an OpenAI-compatible server.

    Parameters
    ----------
    model_path:
        Path to a GGUF model file for llama.cpp.  Mutually exclusive
        with ``base_url``.
    base_url:
        URL of an OpenAI-compatible API server (e.g., Ollama at
        ``http://localhost:11434/v1``).  Mutually exclusive with
        ``model_path``.
    model:
        Model name for the OpenAI-compatible server (e.g., ``"llama3"``).
        Ignored when using ``model_path``.
    n_ctx:
        Context window size for llama.cpp (default 4096).
    n_gpu_layers:
        Number of layers to offload to GPU for llama.cpp (default 0).
    max_tokens:
        Maximum tokens in the response.
    temperature:
        Sampling temperature (0.0-2.0).
    """

    def __init__(
        self,
        model_path: str | None = None,
        base_url: str | None = None,
        model: str = "llama3",
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: object,
    ) -> None:
        if not model_path and not base_url:
            raise ValueError(
                "Either model_path (for llama.cpp) or base_url "
                "(for an OpenAI-compatible server) is required."
            )
        self._model_path = model_path
        self._base_url = base_url
        self._model = model
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._llama: Any = None

    def _load_llama(self) -> Any:
        """Lazily load the llama.cpp model on first use."""
        if self._llama is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise ImportError(
                    "llama-cpp-python is required for LocalLLM with model_path. "
                    "Install it with: pip install intent-engine[local-llm]"
                ) from exc
            logger.info(
                "Loading llama.cpp model from %s (n_ctx=%d, n_gpu_layers=%d)",
                self._model_path,
                self._n_ctx,
                self._n_gpu_layers,
            )
            self._llama = Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )
        return self._llama

    async def interpret(
        self, iml_input: str, context: str | None = None
    ) -> InterpretationResult:
        """Interpret IML-annotated input using a local LLM.

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
            If the model does not reply with the expected JSON.
        """
        system = SYSTEM_PROMPT
        if context:
            system = f"{system}\n\n## Additional Context\n{context}"

        if self._base_url:
            return await self._interpret_via_server(iml_input, system)
        return await self._interpret_via_llama(iml_input, system)

    async def _interpret_via_llama(
        self, iml_input: str, system: str
    ) -> InterpretationResult:
        """Run inference using llama.cpp."""
        llama = self._load_llama()

        response = llama.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": iml_input},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            response_format={"type": "json_object"},
        )

        result = parse_interpretation(
            chat_completion_text(response, "llama.cpp"), "llama.cpp"
        )

        logger.info(
            "llama.cpp interpreted a turn (model=%s, prompt=%s)",
            self._model_path,
            PROMPT_VERSION,
        )
        return result

    async def _interpret_via_server(
        self, iml_input: str, system: str
    ) -> InterpretationResult:
        """Run inference via an OpenAI-compatible local server."""
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for LocalLLM with base_url. "
                "Install it with: pip install intent-engine[openai]"
            ) from exc

        # The client is opened per call: the sync wrappers run every call in a
        # fresh event loop (asyncio.run), and a client keeps its connections
        # bound to the loop that first used it.
        async with AsyncOpenAI(api_key="not-needed", base_url=self._base_url) as client:
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

        result = parse_interpretation(
            chat_completion_text(response, "Local server"), "Local server"
        )

        logger.info(
            "Local server interpreted a turn (model=%s, url=%s, prompt=%s)",
            self._model,
            self._base_url,
            PROMPT_VERSION,
        )
        return result
