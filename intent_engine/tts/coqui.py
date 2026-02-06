"""Coqui TTS adapter.

Uses the Coqui TTS library (open-source, local) to synthesize speech.
Emotion labels are mapped to speaker embeddings or style parameters
depending on the model.  No API key required -- runs on the local machine.
"""

from __future__ import annotations

import io
import logging
import struct
import wave

from intent_engine.tts.base import (
    SynthesisResult,
    TTSProvider,
    get_voice_params,
)

logger = logging.getLogger(__name__)


class CoquiTTS(TTSProvider):
    """Coqui TTS (local) text-to-speech provider.

    Parameters
    ----------
    model_name:
        Coqui TTS model name.  Defaults to
        ``"tts_models/en/ljspeech/tacotron2-DDC"``.
    device:
        Device to run on (``"cpu"`` or ``"cuda"``).
    speaker:
        Speaker name for multi-speaker models, or ``None``.
    language:
        Language code for multi-language models, or ``None``.
    """

    def __init__(
        self,
        model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
        device: str = "cpu",
        speaker: str | None = None,
        language: str | None = None,
        **kwargs: object,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._speaker = speaker
        self._language = language
        self._tts: object | None = None

    def _load_model(self) -> object:
        """Lazily load the Coqui TTS model on first use."""
        if self._tts is None:
            try:
                from TTS.api import TTS  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "TTS (Coqui) is required for CoquiTTS. "
                    "Install it with: pip install intent-engine[coqui]"
                ) from exc
            logger.info(
                "Loading Coqui TTS model '%s' on %s",
                self._model_name,
                self._device,
            )
            self._tts = TTS(model_name=self._model_name).to(self._device)
        return self._tts

    async def synthesize(
        self, text: str, emotion: str = "neutral", **kwargs: object
    ) -> SynthesisResult:
        """Synthesize speech using the local Coqui TTS model.

        Parameters
        ----------
        text:
            Text to synthesize.
        emotion:
            Emotion label used to adjust synthesis parameters
            (speed via voice params).

        Returns
        -------
        SynthesisResult
            Synthesized audio bytes in WAV format.
        """
        tts = self._load_model()
        voice_params = get_voice_params(emotion)

        # Coqui TTS tts() returns a list of float samples
        wav_samples: list[float] = tts.tts(  # type: ignore[union-attr]
            text=text,
            speaker=self._speaker,
            language=self._language,
            speed=voice_params.rate,
        )

        # Convert float samples to 16-bit PCM WAV bytes
        sample_rate = 22050
        audio_bytes = _float_samples_to_wav(wav_samples, sample_rate)

        duration = len(wav_samples) / sample_rate if wav_samples else None

        logger.info(
            "Coqui TTS synthesized %d bytes (model=%s, emotion=%s, duration=%.2fs)",
            len(audio_bytes),
            self._model_name,
            emotion,
            duration or 0.0,
        )

        return SynthesisResult(
            audio_data=audio_bytes,
            format="wav",
            sample_rate=sample_rate,
            duration=duration,
        )


def _float_samples_to_wav(samples: list[float], sample_rate: int) -> bytes:
    """Convert a list of float samples [-1.0, 1.0] to WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # Clamp and convert to 16-bit signed integers
        pcm_data = b"".join(
            struct.pack("<h", max(-32768, min(32767, int(s * 32767))))
            for s in samples
        )
        wf.writeframes(pcm_data)
    return buf.getvalue()
