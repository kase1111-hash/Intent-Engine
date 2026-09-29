"""Abstract TTS provider interface, shared types, and emotion mapping.

All TTS adapters implement ``TTSProvider`` and return
``SynthesisResult`` objects containing raw audio bytes and metadata.
The emotion-to-voice parameter mapping table aligns with the
Prosody Protocol's core emotion vocabulary.
"""

from __future__ import annotations

import html
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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


def normalize_emotion(emotion: object) -> str:
    """Map an emotion value onto the core vocabulary, never raising.

    Labels are matched case-insensitively and ignoring surrounding
    whitespace.  ``None``, an empty string, non-string values and labels
    outside the core vocabulary all resolve to ``"neutral"``; anything
    other than ``None`` or an empty string is logged at warning level so
    a mislabelled emotion does not silently become a neutral voice.

    Parameters
    ----------
    emotion:
        Emotion label, normally from the Prosody Protocol vocabulary but
        possibly whatever an LLM returned.

    Returns
    -------
    str
        A key of :data:`EMOTION_VOICE_MAP`.
    """
    if emotion is None:
        return "neutral"
    if not isinstance(emotion, str):
        logger.warning(
            "Emotion must be a string, got %s; using neutral", type(emotion).__name__
        )
        return "neutral"
    key = emotion.strip().lower()
    if key in EMOTION_VOICE_MAP:
        return key
    if key:
        logger.warning("Unknown emotion %.40r; using neutral", emotion)
    return "neutral"


def get_voice_params(emotion: object) -> EmotionVoiceParams:
    """Look up voice parameters for the given emotion.

    Falls back to ``"neutral"`` if the emotion is not in the core
    vocabulary (see :func:`normalize_emotion`).

    Parameters
    ----------
    emotion:
        Emotion label from the Prosody Protocol vocabulary.

    Returns
    -------
    EmotionVoiceParams
        Voice synthesis parameters for the emotion.
    """
    return EMOTION_VOICE_MAP[normalize_emotion(emotion)]


# Tag bodies exclude "<" so a run of "<" without a ">" cannot make matching quadratic.
_SSML_DOCUMENT_RE = re.compile(
    r"\A\s*(?:<\?xml[^<>]*\?>\s*)?<speak(?:\s[^<>]*)?>.*</speak>\s*\Z", re.DOTALL
)
_SSML_BOUNDARY_TAG_RE = re.compile(r"</?(?:speak|s|p|break|voice)(?:\s[^<>]*)?/?>")
_SSML_TAG_RE = re.compile(r"<[^<>]*>")


def strip_ssml(text: str) -> str:
    """Reduce a complete SSML document to the plain text it speaks.

    Plain-text engines read markup aloud, so adapters that cannot
    interpret SSML pass their input through this first.  Only text that
    is a whole ``<speak>...</speak>`` document is touched; everything
    else, including plain text that merely contains ``<`` or ``>``, is
    returned unchanged.

    Parameters
    ----------
    text:
        Text about to be synthesized.

    Returns
    -------
    str
        The document's text with tags removed and entities decoded, or
        ``text`` itself when it is not an SSML document.
    """
    if not _SSML_DOCUMENT_RE.match(text):
        return text
    spaced = _SSML_BOUNDARY_TAG_RE.sub(" ", text)
    return " ".join(html.unescape(_SSML_TAG_RE.sub("", spaced)).split())


@dataclass(frozen=True)
class SynthesisResult:
    """Output of a TTS provider's ``synthesize()`` call.

    Attributes
    ----------
    audio_data:
        Raw audio bytes.
    format:
        Audio format (e.g., ``"wav"``, ``"mp3"``).  Raw sample formats
        such as ``"pcm"`` and ``"ulaw"`` have no container header.
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

    Attributes
    ----------
    supports_ssml:
        Whether :meth:`synthesize` interprets SSML markup passed as
        ``text``.  ``False`` for every built-in adapter: they treat the
        text as plain text, so callers should send plain text unless a
        provider sets this to ``True``.
    """

    supports_ssml: bool = False

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
