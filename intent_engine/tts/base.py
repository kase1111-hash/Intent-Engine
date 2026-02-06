"""Abstract TTS provider interface, shared types, and emotion mapping.

All TTS adapters implement ``TTSProvider`` and return
``SynthesisResult`` objects containing raw audio bytes and metadata.
The emotion-to-voice parameter mapping table aligns with the
Prosody Protocol's core emotion vocabulary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionVoiceParams:
    """Voice synthesis parameters for a given emotion.

    Attributes
    ----------
    pitch_shift:
        Relative pitch adjustment (e.g., ``"+10%"``, ``"-5%"``).
    rate:
        Speaking rate multiplier (e.g., ``1.15`` = 15% faster).
    volume_db:
        Volume adjustment in decibels (e.g., ``+3``, ``-4``).
    style_notes:
        Human-readable notes on voice quality for this emotion.
    """

    pitch_shift: str
    rate: float
    volume_db: float
    style_notes: str


EMOTION_VOICE_MAP: dict[str, EmotionVoiceParams] = {
    "neutral": EmotionVoiceParams(
        pitch_shift="0%", rate=1.0, volume_db=0.0,
        style_notes="Default baseline",
    ),
    "sincere": EmotionVoiceParams(
        pitch_shift="-2%", rate=0.95, volume_db=0.0,
        style_notes="Warm, genuine",
    ),
    "sarcastic": EmotionVoiceParams(
        pitch_shift="+8%", rate=0.95, volume_db=1.0,
        style_notes="Exaggerated pitch contour",
    ),
    "frustrated": EmotionVoiceParams(
        pitch_shift="+5%", rate=1.1, volume_db=3.0,
        style_notes="Tense, slightly faster",
    ),
    "joyful": EmotionVoiceParams(
        pitch_shift="+10%", rate=1.15, volume_db=2.0,
        style_notes="Bright, upbeat",
    ),
    "uncertain": EmotionVoiceParams(
        pitch_shift="+3%", rate=0.9, volume_db=-1.0,
        style_notes="Rising intonation, hesitant",
    ),
    "angry": EmotionVoiceParams(
        pitch_shift="+5%", rate=1.2, volume_db=6.0,
        style_notes="Tense, fast, loud",
    ),
    "sad": EmotionVoiceParams(
        pitch_shift="-8%", rate=0.8, volume_db=-4.0,
        style_notes="Lower, slower, quiet",
    ),
    "fearful": EmotionVoiceParams(
        pitch_shift="+6%", rate=1.15, volume_db=-2.0,
        style_notes="Higher pitch, fast, quiet",
    ),
    "surprised": EmotionVoiceParams(
        pitch_shift="+12%", rate=1.1, volume_db=2.0,
        style_notes="Sharp rise, wide pitch range",
    ),
    "disgusted": EmotionVoiceParams(
        pitch_shift="-3%", rate=0.9, volume_db=1.0,
        style_notes="Low, creaky, slow",
    ),
    "calm": EmotionVoiceParams(
        pitch_shift="0%", rate=0.95, volume_db=0.0,
        style_notes="Even, measured",
    ),
    "empathetic": EmotionVoiceParams(
        pitch_shift="-5%", rate=0.9, volume_db=-2.0,
        style_notes="Warm, slightly slower",
    ),
}
"""Emotion-to-voice parameter mapping aligned with Prosody Protocol core vocabulary."""


def get_voice_params(emotion: str) -> EmotionVoiceParams:
    """Look up voice parameters for the given emotion.

    Falls back to ``"neutral"`` if the emotion is not in the core vocabulary.

    Parameters
    ----------
    emotion:
        Emotion label from the Prosody Protocol vocabulary.

    Returns
    -------
    EmotionVoiceParams
        Voice synthesis parameters for the emotion.
    """
    return EMOTION_VOICE_MAP.get(emotion, EMOTION_VOICE_MAP["neutral"])


@dataclass(frozen=True)
class SynthesisResult:
    """Output of a TTS provider's ``synthesize()`` call.

    Attributes
    ----------
    audio_data:
        Raw audio bytes.
    format:
        Audio format (e.g., ``"wav"``, ``"mp3"``).
    sample_rate:
        Sample rate in Hz.
    duration:
        Duration in seconds, or ``None`` if unknown.
    """

    audio_data: bytes
    format: str = "wav"
    sample_rate: int = 22050
    duration: float | None = None


class TTSProvider(ABC):
    """Abstract base class for text-to-speech provider adapters.

    Subclasses must implement :meth:`synthesize` which converts text
    and an emotion label into audio bytes. The orchestrator
    (``IntentEngine``) wraps the result in an ``Audio`` object.
    """

    @abstractmethod
    async def synthesize(
        self, text: str, emotion: str = "neutral", **kwargs: object
    ) -> SynthesisResult:
        """Synthesize speech with emotional tone.

        Parameters
        ----------
        text:
            The text to synthesize.
        emotion:
            Emotion label from the Prosody Protocol core vocabulary
            (e.g., ``"empathetic"``, ``"frustrated"``).
        **kwargs:
            Provider-specific parameters.

        Returns
        -------
        SynthesisResult
            Raw audio bytes and metadata.

        Raises
        ------
        intent_engine.errors.TTSError
            If synthesis fails.
        """
        ...
