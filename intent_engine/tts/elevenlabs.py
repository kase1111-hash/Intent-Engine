"""ElevenLabs TTS adapter.

Uses the ElevenLabs API to synthesize speech with emotional voice settings.
Emotion labels are mapped to ElevenLabs voice parameter adjustments
(stability, similarity_boost, style, use_speaker_boost).
Requires an ``ELEVENLABS_API_KEY`` environment variable or explicit
``api_key`` parameter.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os

from intent_engine.tts.base import (
    EMOTION_VOICE_MAP,
    SynthesisResult,
    TTSProvider,
    get_voice_params,
)

logger = logging.getLogger(__name__)

# Map Prosody Protocol core emotions to ElevenLabs voice settings.
# stability: 0.0 (more variable) - 1.0 (more stable)
# similarity_boost: 0.0 (more diverse) - 1.0 (closer to original voice)
# style: 0.0 (neutral style) - 1.0 (exaggerated style)
ELEVENLABS_EMOTION_SETTINGS: dict[str, dict[str, float]] = {
    "neutral":     {"stability": 0.50, "similarity_boost": 0.75, "style": 0.0},
    "sincere":     {"stability": 0.60, "similarity_boost": 0.80, "style": 0.2},
    "sarcastic":   {"stability": 0.30, "similarity_boost": 0.60, "style": 0.8},
    "frustrated":  {"stability": 0.35, "similarity_boost": 0.70, "style": 0.6},
    "joyful":      {"stability": 0.40, "similarity_boost": 0.75, "style": 0.7},
    "uncertain":   {"stability": 0.30, "similarity_boost": 0.70, "style": 0.4},
    "angry":       {"stability": 0.25, "similarity_boost": 0.65, "style": 0.9},
    "sad":         {"stability": 0.55, "similarity_boost": 0.80, "style": 0.5},
    "fearful":     {"stability": 0.25, "similarity_boost": 0.70, "style": 0.6},
    "surprised":   {"stability": 0.30, "similarity_boost": 0.65, "style": 0.7},
    "disgusted":   {"stability": 0.45, "similarity_boost": 0.70, "style": 0.5},
    "calm":        {"stability": 0.70, "similarity_boost": 0.80, "style": 0.1},
    "empathetic":  {"stability": 0.60, "similarity_boost": 0.85, "style": 0.3},
}


class ElevenLabsTTS(TTSProvider):
    """ElevenLabs API-based text-to-speech provider.

    Parameters
    ----------
    api_key:
        ElevenLabs API key.  Falls back to the ``ELEVENLABS_API_KEY``
        environment variable if not provided.
    voice_id:
        ElevenLabs voice ID to use.  Defaults to ``"21m00Tcm4TlvDq8ikWAM"``
        (Rachel).
    model_id:
        ElevenLabs model to use.  Defaults to ``"eleven_monolingual_v1"``.
    output_format:
        Audio output format.  Defaults to ``"mp3_44100_128"``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_monolingual_v1",
        output_format: str = "mp3_44100_128",
        **kwargs: object,
    ) -> None:
        self._api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "ElevenLabs API key is required. Set ELEVENLABS_API_KEY or pass api_key=."
            )
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = output_format
        self._client: object | None = None

    async def synthesize(
        self, text: str, emotion: str = "neutral", **kwargs: object
    ) -> SynthesisResult:
        """Synthesize speech using the ElevenLabs API.

        Parameters
        ----------
        text:
            Text to synthesize.
        emotion:
            Emotion label for voice parameter adjustment.

        Returns
        -------
        SynthesisResult
            Synthesized audio bytes and metadata.
        """
        if self._client is None:
            try:
                from elevenlabs import ElevenLabs as ElevenLabsClient  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "elevenlabs is required for ElevenLabsTTS. "
                    "Install it with: pip install intent-engine[elevenlabs]"
                ) from exc
            self._client = ElevenLabsClient(api_key=self._api_key)

        settings = ELEVENLABS_EMOTION_SETTINGS.get(
            emotion, ELEVENLABS_EMOTION_SETTINGS["neutral"]
        )

        # Run the synchronous ElevenLabs client in a thread executor
        # to avoid blocking the event loop.
        loop = asyncio.get_running_loop()
        audio_iterator = await loop.run_in_executor(
            None,
            functools.partial(
                self._client.text_to_speech.convert,  # type: ignore[union-attr]
                voice_id=self._voice_id,
                text=text,
                model_id=self._model_id,
                output_format=self._output_format,
                voice_settings={
                    "stability": settings["stability"],
                    "similarity_boost": settings["similarity_boost"],
                    "style": settings["style"],
                    "use_speaker_boost": True,
                },
            ),
        )

        # Collect streamed audio chunks into a single bytes object
        audio_bytes = b"".join(audio_iterator)

        audio_format = "mp3" if "mp3" in self._output_format else "wav"
        sample_rate = 44100 if "44100" in self._output_format else 22050

        logger.info(
            "ElevenLabs synthesized %d bytes (voice=%s, emotion=%s, format=%s)",
            len(audio_bytes),
            self._voice_id,
            emotion,
            audio_format,
        )

        return SynthesisResult(
            audio_data=audio_bytes,
            format=audio_format,
            sample_rate=sample_rate,
        )
