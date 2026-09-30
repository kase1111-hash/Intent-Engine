"""ElevenLabs TTS adapter.

Uses the ElevenLabs API to synthesize speech with emotional voice settings.
Emotion labels are mapped to ElevenLabs voice parameter adjustments
(stability, similarity_boost, style, use_speaker_boost).
Requires an ``ELEVENLABS_API_KEY`` environment variable or explicit
``api_key`` parameter.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

from intent_engine.tts.base import (
    SynthesisResult,
    TTSProvider,
    normalize_emotion,
)

logger = logging.getLogger(__name__)

# ElevenLabs names output formats "<codec>_<sample rate>[_<bitrate>]",
# e.g. "mp3_44100_128", "pcm_16000", "ulaw_8000".
_OUTPUT_FORMAT_RE = re.compile(r"([a-z0-9]+)_(\d+)(?:_\d+)?")

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


def _parse_output_format(output_format: str) -> tuple[str, int]:
    """Return the ``(format, sample_rate)`` an ElevenLabs output format yields.

    The format is the codec name (``"mp3"``, ``"pcm"``, ``"ulaw"``,
    ``"alaw"``, ``"opus"`` or ``"wav"``); ``pcm``, ``ulaw`` and ``alaw`` are
    raw samples without a container.  A name that does not follow the
    ``<codec>_<sample rate>[_<bitrate>]`` pattern is reported as its leading
    part at 44100 Hz, ElevenLabs' default rate, after a warning.
    """
    match = _OUTPUT_FORMAT_RE.fullmatch(output_format)
    if match is None:
        codec = output_format.split("_")[0] or "mp3"
        logger.warning(
            "Cannot read a sample rate from output_format %r; reporting %s at 44100 Hz",
            output_format,
            codec,
        )
        return codec, 44100
    return match[1], int(match[2])


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
        Audio output format, named ``<codec>_<sample rate>[_<bitrate>]``.
        Defaults to ``"mp3_44100_128"``.  The result reports the codec as
        its ``format`` (``pcm``, ``ulaw`` and ``alaw`` are raw samples) and
        the sample rate from the name.
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
        self._audio_format, self._sample_rate = _parse_output_format(output_format)
        self._client: Any = None

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
                from elevenlabs import ElevenLabs as ElevenLabsClient
            except ImportError as exc:
                raise ImportError(
                    "elevenlabs is required for ElevenLabsTTS. "
                    "Install it with: pip install intent-engine[elevenlabs]"
                ) from exc
            self._client = ElevenLabsClient(api_key=self._api_key)

        settings = ELEVENLABS_EMOTION_SETTINGS[normalize_emotion(emotion)]
        client = self._client

        def convert() -> bytes:
            # The SDK's convert() is a generator: the HTTP request only happens
            # while the chunks are consumed, so consume them here, in the worker
            # thread, rather than on the event loop.
            return b"".join(
                client.text_to_speech.convert(
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
                )
            )

        audio_bytes = await asyncio.to_thread(convert)

        # The emotion is not logged above DEBUG level: emotional data is sensitive.
        logger.info(
            "ElevenLabs synthesized %d bytes (voice=%s, format=%s)",
            len(audio_bytes),
            self._voice_id,
            self._audio_format,
        )

        return SynthesisResult(
            audio_data=audio_bytes,
            format=self._audio_format,
            sample_rate=self._sample_rate,
        )
