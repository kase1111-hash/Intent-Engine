"""HybridEngine -- cloud STT/TTS + local LLM deployment.

Uses cloud providers for STT and TTS (quality benefit) while
running the LLM locally for privacy, data sovereignty, and no
per-request cost.  Prosody analysis via ``prosody_protocol``
runs locally by default.
"""

from __future__ import annotations

import logging
from typing import Any

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
        Path to a local GGUF model file or Ollama model name.
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
    """

    def __init__(
        self,
        stt_provider: str = _HYBRID_DEFAULTS["stt_provider"],
        llm_provider: str = _HYBRID_DEFAULTS["llm_provider"],
        tts_provider: str = _HYBRID_DEFAULTS["tts_provider"],
        llm_model: str | None = None,
        constitutional_rules: str | None = None,
        prosody_profile: str | None = None,
        cache_size: int = 128,
        stt_kwargs: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        tts_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # Merge llm_model into llm_kwargs if provided
        llm_kw = dict(llm_kwargs or {})
        if llm_model:
            llm_kw["model_path"] = llm_model

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
        """Whether the LLM runs locally."""
        return True

    @property
    def deployment_mode(self) -> str:
        """Return the deployment mode identifier."""
        return "hybrid"
