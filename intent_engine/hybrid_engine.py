"""HybridEngine -- cloud STT/TTS + local LLM deployment.

Uses cloud providers for STT and TTS (quality benefit) while
running the LLM locally for privacy, data sovereignty, and no
per-request cost.  Prosody analysis via ``prosody_protocol``
runs locally by default.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from intent_engine._deployment import (
    is_local_llm,
    llama_cpp_model_file,
    llm_kwargs_with_model,
)
from intent_engine.engine import IntentEngine

logger = logging.getLogger(__name__)

# Default provider selections for hybrid mode
_HYBRID_DEFAULTS = {
    "stt_provider": "deepgram",
    "llm_provider": "local",
    "tts_provider": "coqui",
}


class HybridEngine(IntentEngine):
    """Hybrid deployment: cloud STT/TTS with local LLM.

    Inherits the full ``IntentEngine`` pipeline but selects
    providers that balance cloud quality with local privacy:

    - **STT**: Cloud (Deepgram by default) for transcription quality
    - **LLM**: Local (llama.cpp / Ollama) for privacy and sovereignty
    - **TTS**: Local (Coqui by default) for low latency
    - **Prosody analysis**: Always local via ``prosody_protocol``

    Parameters
    ----------
    stt_provider:
        Cloud STT provider (``"deepgram"`` or ``"assemblyai"``).
    llm_provider:
        Local LLM provider (``"local"``).
    tts_provider:
        TTS provider (``"coqui"``, ``"espeak"``, or ``"elevenlabs"``).
    llm_model:
        Path to a local GGUF model file (llama.cpp) or, with
        ``llm_kwargs={"base_url": ...}``, a model name on a local
        OpenAI-compatible server such as Ollama or vLLM.  For a cloud LLM
        provider it is that provider's model name.
    constitutional_rules:
        Optional path to a YAML file with constitutional rules.
    prosody_profile:
        Optional path to a prosody profile JSON.
    cache_size:
        Maximum number of audio results to cache.
    stt_kwargs:
        Additional keyword arguments for the STT adapter.
    llm_kwargs:
        Additional keyword arguments for the LLM adapter.
    tts_kwargs:
        Additional keyword arguments for the TTS adapter.
    validate_models:
        If ``True`` (default), raise ``FileNotFoundError`` when the llama.cpp
        model file (``llm_model``, or a ``model_path`` in ``llm_kwargs``)
        does not exist; a leading ``~`` is the home directory.  Model names
        for a server or a cloud provider are not files and are not checked.
    """

    def __init__(
        self,
        stt_provider: str = _HYBRID_DEFAULTS["stt_provider"],
        llm_provider: str = _HYBRID_DEFAULTS["llm_provider"],
        tts_provider: str = _HYBRID_DEFAULTS["tts_provider"],
        llm_model: str | None = None,
        constitutional_rules: str | os.PathLike[str] | None = None,
        prosody_profile: str | os.PathLike[str] | None = None,
        cache_size: int = 128,
        stt_kwargs: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        tts_kwargs: dict[str, Any] | None = None,
        validate_models: bool = True,
    ) -> None:
        # Hand the model to the parameter the LLM provider really takes
        llm_kw = llm_kwargs_with_model(llm_provider, llm_model, llm_kwargs)

        # Fail before any cloud STT is paid for: a llama.cpp file that is not there
        model_file = llama_cpp_model_file(llm_provider, llm_kw)
        if validate_models and model_file is not None and not Path(model_file).exists():
            source = "llm_model" if llm_model else "llm_kwargs model_path"
            raise FileNotFoundError(f"{source} path does not exist: {model_file}")

        super().__init__(
            stt_provider=stt_provider,
            llm_provider=llm_provider,
            tts_provider=tts_provider,
            constitutional_rules=constitutional_rules,
            prosody_profile=prosody_profile,
            cache_size=cache_size,
            stt_kwargs=stt_kwargs,
            llm_kwargs=llm_kw,
            tts_kwargs=tts_kwargs,
        )

        self._llm_model = llm_model
        self._is_llm_local = is_local_llm(llm_provider, llm_kw)
        if not self._is_llm_local:
            logger.warning(
                "HybridEngine LLM is not local (llm=%s): prompts, including "
                "transcripts, may leave this machine or network",
                llm_provider,
            )

        logger.info(
            "HybridEngine initialized (stt=%s [cloud], llm=%s [local], tts=%s, model=%s)",
            stt_provider,
            llm_provider,
            tts_provider,
            llm_model or "default",
        )

    @property
    def llm_model(self) -> str | None:
        """Path or name of the local LLM model."""
        return self._llm_model

    @property
    def is_llm_local(self) -> bool:
        """Whether the LLM runs on this machine or a private network.

        ``False`` for a cloud LLM provider, or a ``local`` server whose
        ``base_url`` is a public address (see ``is_local_url``).
        """
        return self._is_llm_local

    @property
    def deployment_mode(self) -> str:
        """Return the deployment mode identifier (``"hybrid"``)."""
        return "hybrid"
