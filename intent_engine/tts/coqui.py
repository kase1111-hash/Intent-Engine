"""Coqui TTS adapter.

Uses the Coqui TTS library (open-source, local) to synthesize speech.
The emotion label only sets the ``speed`` passed to ``TTS.tts()``; whether
that changes the voice depends on the model and library version (see
:class:`CoquiTTS`).  No API key required -- runs on the local machine.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import struct
import threading
import wave
from typing import Any

from intent_engine.tts.base import (
    SynthesisResult,
    TTSProvider,
    get_voice_params,
    strip_ssml,
)

logger = logging.getLogger(__name__)

# Rate of most Coqui models, and the fallback when the loaded one does not report it.
_DEFAULT_SAMPLE_RATE = 22050


def _output_sample_rate(tts: Any) -> int:
    """Return the rate the loaded model synthesizes at (models differ: XTTS is 24000)."""
    rate = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", None)
    if (
        isinstance(rate, (int, float))
        and not isinstance(rate, bool)
        and math.isfinite(rate)
        and rate > 0
    ):
        return int(rate)
    return _DEFAULT_SAMPLE_RATE


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

    Notes
    -----
    Coqui has no emotion control, so the emotion label is mapped to the
    ``speed`` argument of ``TTS.tts()`` only.  According to the sources of
    TTS 0.22.0 and coqui-tts 0.27.5: 0.22.0 accepts ``speed`` but discards it
    for every model; coqui-tts forwards it to the model, where XTTS applies
    it and other models (including the default Tacotron2) ignore it.  With
    those, every emotion sounds the same.

    Both packages are imported as ``TTS``.  TTS 0.22.0, the last release of
    the original package, does not install on Python 3.12 or later; the
    maintained fork ``coqui-tts`` does.
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
        self._tts: Any = None
        # Coqui models are not documented as thread-safe: load and run one call at a time.
        self._lock = threading.Lock()

    def _load_model(self) -> Any:
        """Lazily load the Coqui TTS model on first use."""
        if self._tts is None:
            try:
                from TTS.api import TTS
            except ImportError as exc:
                raise ImportError(
                    "TTS (Coqui) is required for CoquiTTS. "
                    "Install it with: pip install intent-engine[coqui] "
                    "(Python < 3.12), or pip install coqui-tts, the maintained "
                    "fork, on newer Pythons."
                ) from exc
            logger.info(
                "Loading Coqui TTS model '%s' on %s",
                self._model_name,
                self._device,
            )
            self._tts = TTS(model_name=self._model_name).to(self._device)
        return self._tts

    def _synthesize_blocking(self, text: str, speed: float) -> SynthesisResult:
        """Load the model if needed and synthesize (runs in a worker thread)."""
        with self._lock:
            tts = self._load_model()

            # Coqui TTS tts() returns a list of float samples
            wav_samples: list[float] = tts.tts(
                text=text,
                speaker=self._speaker,
                language=self._language,
                speed=speed,
            )
            sample_rate = _output_sample_rate(tts)

        # Convert float samples to 16-bit PCM WAV bytes
        audio_bytes = _float_samples_to_wav(wav_samples, sample_rate)

        duration = len(wav_samples) / sample_rate if len(wav_samples) else None

        return SynthesisResult(
            audio_data=audio_bytes,
            format="wav",
            sample_rate=sample_rate,
            duration=duration,
        )

    async def synthesize(
        self, text: str, emotion: str = "neutral", **kwargs: object
    ) -> SynthesisResult:
        """Synthesize speech using the local Coqui TTS model.

        Parameters
        ----------
        text:
            Text to synthesize.  An SSML document is reduced to its plain
            text first.
        emotion:
            Emotion label used to adjust synthesis parameters
            (speed via voice params; see the class notes for its limits).

        Returns
        -------
        SynthesisResult
            Synthesized audio bytes in WAV format, at the sample rate of
            the loaded model.
        """
        voice_params = get_voice_params(emotion)

        # Model loading and inference are CPU/GPU-bound and blocking, so keep
        # them off the event loop.
        result = await asyncio.to_thread(
            self._synthesize_blocking, strip_ssml(text), voice_params.rate
        )

        logger.info(
            "Coqui TTS synthesized %d bytes (model=%s, emotion=%s, duration=%.2fs)",
            len(result.audio_data),
            self._model_name,
            emotion,
            result.duration or 0.0,
        )

        return result


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
