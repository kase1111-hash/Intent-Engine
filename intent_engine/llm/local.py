"""Local LLM adapter -- llama.cpp / vLLM / Ollama.

Supports two backends:

1. **llama.cpp** via ``llama-cpp-python``: Load a GGUF model file
   directly and run inference locally.
2. **OpenAI-compatible server** (vLLM, Ollama, etc.): Connect to a
   local HTTP endpoint that exposes an OpenAI-compatible chat API.

No external API key required -- everything runs on the local machine.
"""

from __future__ import annotations

import json
import logging

from intent_engine.llm.base import InterpretationResult, LLMProvider
from intent_engine.llm.prompts import SYSTEM_PROMPT

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
        self._llama: object | None = None

    def _load_llama(self) -> object:
        """Lazily load the llama.cpp model on first use."""
        if self._llama is None:
            try:
                from llama_cpp import Llama  # type: ignore[import-untyped]
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

        response = llama.create_chat_completion(  # type: ignore[union-attr]
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": iml_input},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            response_format={"type": "json_object"},
        )

        raw_text = response["choices"][0]["message"]["content"] or ""  # type: ignore[index]
        parsed = json.loads(raw_text)

        logger.info(
            "llama.cpp interpreted intent=%s emotion=%s (model=%s)",
            parsed.get("intent"),
            parsed.get("suggested_emotion"),
            self._model_path,
        )

        return InterpretationResult(
            intent=parsed["intent"],
            response_text=parsed["response_text"],
            suggested_emotion=parsed["suggested_emotion"],
        )

    async def _interpret_via_server(
        self, iml_input: str, system: str
    ) -> InterpretationResult:
        """Run inference via an OpenAI-compatible local server."""
        try:
            from openai import AsyncOpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "openai is required for LocalLLM with base_url. "
                "Install it with: pip install intent-engine[openai]"
            ) from exc

        client = AsyncOpenAI(api_key="not-needed", base_url=self._base_url)

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

        raw_text = response.choices[0].message.content or ""
        parsed = json.loads(raw_text)

        logger.info(
            "Local server interpreted intent=%s emotion=%s (model=%s, url=%s)",
            parsed.get("intent"),
            parsed.get("suggested_emotion"),
            self._model,
            self._base_url,
        )

        return InterpretationResult(
            intent=parsed["intent"],
            response_text=parsed["response_text"],
            suggested_emotion=parsed["suggested_emotion"],
        )
