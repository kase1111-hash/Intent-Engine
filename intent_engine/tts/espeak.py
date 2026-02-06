"""eSpeak TTS adapter.

Uses ``pyttsx3`` (which wraps eSpeak on Linux) for lightweight,
open-source text-to-speech synthesis.  Optionally uses
``prosody_protocol.IMLToSSML`` to convert emotion parameters into
SSML markup for enhanced prosodic control.

No API key required -- runs entirely on the local machine.
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

from intent_engine.tts.base import (
    SynthesisResult,
    TTSProvider,
    get_voice_params,
)

logger = logging.getLogger(__name__)


class ESpeakTTS(TTSProvider):
    """eSpeak / pyttsx3-based text-to-speech provider.

    Parameters
    ----------
    voice:
        Voice name or ID to use (e.g., ``"english"``, ``"english+f3"``).
        If ``None``, uses the system default.
    rate_wpm:
        Base speaking rate in words per minute.  Defaults to ``175``.
    volume:
        Base volume (0.0 - 1.0).  Defaults to ``1.0``.
    """

    def __init__(
        self,
        voice: str | None = None,
        rate_wpm: int = 175,
        volume: float = 1.0,
        **kwargs: object,
    ) -> None:
        self._voice = voice
        self._rate_wpm = rate_wpm
        self._volume = volume

    def _create_engine(self) -> object:
        """Create a new pyttsx3 engine instance."""
        try:
            import pyttsx3  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pyttsx3 is required for ESpeakTTS. "
                "Install it with: pip install intent-engine[espeak]"
            ) from exc

        engine = pyttsx3.init()

        if self._voice:
            engine.setProperty("voice", self._voice)

        return engine

    async def synthesize(
        self, text: str, emotion: str = "neutral", **kwargs: object
    ) -> SynthesisResult:
        """Synthesize speech using eSpeak via pyttsx3.

        Parameters
        ----------
        text:
            Text to synthesize.
        emotion:
            Emotion label used to adjust rate and volume.

        Returns
        -------
        SynthesisResult
            Synthesized audio bytes in WAV format.
        """
        engine = self._create_engine()
        voice_params = get_voice_params(emotion)

        # Adjust rate and volume based on emotion
        adjusted_rate = int(self._rate_wpm * voice_params.rate)
        adjusted_volume = min(1.0, max(0.0, self._volume + voice_params.volume_db / 20.0))

        engine.setProperty("rate", adjusted_rate)
        engine.setProperty("volume", adjusted_volume)

        # pyttsx3 can only save to a file, so use a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()

            audio_bytes = Path(tmp_path).read_bytes()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        logger.info(
            "eSpeak synthesized %d bytes (emotion=%s, rate=%d wpm, volume=%.2f)",
            len(audio_bytes),
            emotion,
            adjusted_rate,
            adjusted_volume,
        )

        return SynthesisResult(
            audio_data=audio_bytes,
            format="wav",
            sample_rate=22050,
        )
