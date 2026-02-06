"""TTS provider adapters.

Provides a provider-agnostic interface for emotionally-aware speech
synthesis. All adapters accept text plus an emotion label from the
Prosody Protocol core vocabulary and return ``SynthesisResult`` objects.

Usage::

    from intent_engine.tts import create_tts_provider

    tts = create_tts_provider("elevenlabs", api_key="...")
    result = await tts.synthesize("Hello!", emotion="joyful")
"""

from __future__ import annotations

from typing import Any

from intent_engine.tts.base import (
    EMOTION_VOICE_MAP,
    EmotionVoiceParams,
    SynthesisResult,
    TTSProvider,
    get_voice_params,
)

__all__ = [
    "TTSProvider",
    "SynthesisResult",
    "EmotionVoiceParams",
    "EMOTION_VOICE_MAP",
    "get_voice_params",
    "TTS_PROVIDERS",
    "create_tts_provider",
]


def _get_provider_class(name: str) -> type[TTSProvider]:
    """Lazily import provider classes to avoid requiring all SDKs at once."""
    if name == "elevenlabs":
        from intent_engine.tts.elevenlabs import ElevenLabsTTS

        return ElevenLabsTTS
    if name == "coqui":
        from intent_engine.tts.coqui import CoquiTTS

        return CoquiTTS
    if name == "espeak":
        from intent_engine.tts.espeak import ESpeakTTS

        return ESpeakTTS
    raise ValueError(
        f"Unknown TTS provider: {name!r}. "
        f"Available providers: {', '.join(TTS_PROVIDERS)}"
    )


TTS_PROVIDERS: dict[str, str] = {
    "elevenlabs": "intent_engine.tts.elevenlabs.ElevenLabsTTS",
    "coqui": "intent_engine.tts.coqui.CoquiTTS",
    "espeak": "intent_engine.tts.espeak.ESpeakTTS",
}
"""Registry of available TTS provider names."""


def create_tts_provider(name: str, **kwargs: Any) -> TTSProvider:
    """Create a TTS provider instance by name.

    Parameters
    ----------
    name:
        Provider name.  One of ``"elevenlabs"``, ``"coqui"``, ``"espeak"``.
    **kwargs:
        Provider-specific configuration passed to the constructor.

    Returns
    -------
    TTSProvider
        An initialized TTS provider instance.
    """
    cls = _get_provider_class(name)
    return cls(**kwargs)
