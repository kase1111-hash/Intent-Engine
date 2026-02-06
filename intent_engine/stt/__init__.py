"""STT provider adapters.

Provides a provider-agnostic interface for speech-to-text transcription.
All adapters produce ``prosody_protocol.WordAlignment`` timestamps.

Usage::

    from intent_engine.stt import create_stt_provider

    stt = create_stt_provider("whisper-prosody", model_size="base")
    result = await stt.transcribe("audio.wav")
"""

from __future__ import annotations

from typing import Any

from intent_engine.stt.base import STTProvider, TranscriptionResult

__all__ = [
    "STTProvider",
    "TranscriptionResult",
    "STT_PROVIDERS",
    "create_stt_provider",
]


def _get_provider_class(name: str) -> type[STTProvider]:
    """Lazily import provider classes to avoid requiring all SDKs at once."""
    if name == "whisper-prosody":
        from intent_engine.stt.whisper import WhisperSTT

        return WhisperSTT
    if name == "deepgram":
        from intent_engine.stt.deepgram import DeepgramSTT

        return DeepgramSTT
    if name == "assemblyai":
        from intent_engine.stt.assemblyai import AssemblyAISTT

        return AssemblyAISTT
    raise ValueError(
        f"Unknown STT provider: {name!r}. "
        f"Available providers: {', '.join(STT_PROVIDERS)}"
    )


STT_PROVIDERS: dict[str, str] = {
    "whisper-prosody": "intent_engine.stt.whisper.WhisperSTT",
    "deepgram": "intent_engine.stt.deepgram.DeepgramSTT",
    "assemblyai": "intent_engine.stt.assemblyai.AssemblyAISTT",
}
"""Registry of available STT provider names."""


def create_stt_provider(name: str, **kwargs: Any) -> STTProvider:
    """Create an STT provider instance by name.

    Parameters
    ----------
    name:
        Provider name.  One of ``"whisper-prosody"``, ``"deepgram"``,
        ``"assemblyai"``.
    **kwargs:
        Provider-specific configuration passed to the constructor.

    Returns
    -------
    STTProvider
        An initialized STT provider instance.
    """
    cls = _get_provider_class(name)
    return cls(**kwargs)
