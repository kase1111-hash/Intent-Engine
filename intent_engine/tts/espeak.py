"""eSpeak TTS adapter.

Uses ``pyttsx3`` (which wraps eSpeak on Linux) for lightweight,
open-source text-to-speech synthesis.  The emotion label adjusts the
speaking rate and volume.  ``pyttsx3`` cannot hand SSML to eSpeak, so an
SSML document passed as ``text`` is reduced to the plain text it speaks
rather than being read aloud tag by tag.

No API key required -- runs entirely on the local machine.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from intent_engine.errors import TTSError
from intent_engine.tts.base import (
    EMOTION_VOICE_MAP,
    SynthesisResult,
    TTSProvider,
    get_voice_params,
    strip_ssml,
)

logger = logging.getLogger(__name__)

# eSpeak keeps one process-wide synthesis callback, so only one pyttsx3
# engine can synthesise at a time.
_ENGINE_LOCK = threading.Lock()

# pyttsx3 2.99 returns from runAndWait() as soon as eSpeak has started
# synthesising and writes the file from eSpeak's own thread when it is done,
# so the adapter waits for that.  Generous, because it only matters when the
# engine has stopped responding.
_TIMEOUT_BASE_S = 5.0
_TIMEOUT_PER_CHAR_S = 0.01

_WAV_HEADER_BYTES = 44


def _emotion_volume(volume: float, volume_db: float) -> float:
    """Engine volume (0.0 - 1.0) for an emotion's decibel offset.

    ``volume`` is the level of the loudest emotion in the table; the others
    sit below it by the difference in decibels.  Anchoring the loudest
    emotion at the ceiling leaves headroom for every louder-than-neutral
    emotion; anchoring neutral there would clamp them all to neutral.
    """
    loudest_db = max(params.volume_db for params in EMOTION_VOICE_MAP.values())
    return min(1.0, max(0.0, volume * 10 ** ((volume_db - loudest_db) / 20)))


def _pyttsx3_version() -> tuple[int, int] | None:
    """Installed pyttsx3 ``(major, minor)`` version, or ``None`` if unknown."""
    try:
        raw = importlib.metadata.version("pyttsx3")
    except importlib.metadata.PackageNotFoundError:
        return None
    match = re.match(r"(\d+)\.(\d+)", raw)
    return (int(match[1]), int(match[2])) if match else None


def _is_complete_wav(path: str) -> bool:
    """Whether ``path`` holds a WAV file that has been written out in full."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return False
    return (
        len(data) > _WAV_HEADER_BYTES
        and data[:4] == b"RIFF"
        and data[8:12] == b"WAVE"
        and int.from_bytes(data[4:8], "little") + 8 == len(data)
    )


def _no_audio_error(errors: list[object]) -> TTSError:
    """Build the error for an engine that finished without writing audio."""
    message = "eSpeak produced no audio."
    version = _pyttsx3_version()
    if version is not None and version < (2, 99) and shutil.which("ffmpeg") is None:
        message += (
            " pyttsx3 before 2.99 writes WAV files through the ffmpeg binary, which"
            " was not found on PATH: install ffmpeg or upgrade with"
            " `pip install 'pyttsx3>=2.99'`."
        )
    else:
        message += " Check that eSpeak (espeak-ng) is installed and the voice exists."
    if errors:
        message += f" The engine reported: {errors[-1]}"
    return TTSError(message)


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
        Volume of the loudest emotion (0.0 - 1.0).  Defaults to ``1.0``.
        Other emotions are quieter by the difference in their decibel
        offsets, so neutral speech is below this level.
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

    def _create_engine(self) -> Any:
        """Create a new pyttsx3 engine instance."""
        try:
            import pyttsx3
        except ImportError as exc:
            raise ImportError(
                "pyttsx3 is required for ESpeakTTS. "
                "Install it with: pip install intent-engine[espeak]"
            ) from exc

        engine = pyttsx3.init()

        if self._voice:
            engine.setProperty("voice", self._voice)

        return engine

    def _run_engine(self, engine: Any, text: str, path: str) -> list[object]:
        """Save ``text`` to ``path`` and wait for the engine to finish.

        Returns the errors the engine reported along the way.
        """
        finished = threading.Event()
        errors: list[object] = []

        def on_finished(**_: object) -> None:
            finished.set()

        def on_error(exception: object = None, **_: object) -> None:
            errors.append(exception)

        tokens = [
            engine.connect("finished-utterance", on_finished),
            engine.connect("error", on_error),
        ]
        try:
            engine.save_to_file(text, path)
            engine.runAndWait()

            timeout = _TIMEOUT_BASE_S + _TIMEOUT_PER_CHAR_S * len(text)
            deadline = time.monotonic() + timeout
            while not (finished.is_set() or _is_complete_wav(path)):
                if time.monotonic() >= deadline:
                    message = f"eSpeak timed out after {timeout:g}s waiting for audio."
                    if errors:
                        message += f" The engine reported: {errors[-1]}"
                    raise TTSError(message)
                finished.wait(0.01)
        finally:
            for token in tokens:
                engine.disconnect(token)
        return errors

    def _synthesize_blocking(self, text: str, emotion: str) -> bytes:
        """Synthesize ``text`` and return the WAV bytes (runs in a worker thread)."""
        voice_params = get_voice_params(emotion)

        # Adjust rate and volume based on emotion
        adjusted_rate = int(self._rate_wpm * voice_params.rate)
        adjusted_volume = _emotion_volume(self._volume, voice_params.volume_db)

        # pyttsx3 can only save to a file, so use a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with _ENGINE_LOCK:
                engine = self._create_engine()
                engine.setProperty("rate", adjusted_rate)
                engine.setProperty("volume", adjusted_volume)
                errors = self._run_engine(engine, text, tmp_path)

            try:
                audio_bytes = Path(tmp_path).read_bytes()
            except FileNotFoundError:
                audio_bytes = b""
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if len(audio_bytes) <= _WAV_HEADER_BYTES:
            raise _no_audio_error(errors)
        if errors:
            logger.warning("pyttsx3 reported an error while synthesising: %s", errors[-1])

        logger.info(
            "eSpeak synthesized %d bytes (emotion=%s, rate=%d wpm, volume=%.2f)",
            len(audio_bytes),
            emotion,
            adjusted_rate,
            adjusted_volume,
        )
        return audio_bytes

    async def synthesize(
        self, text: str, emotion: str = "neutral", **kwargs: object
    ) -> SynthesisResult:
        """Synthesize speech using eSpeak via pyttsx3.

        Parameters
        ----------
        text:
            Text to synthesize.  An SSML document is reduced to its plain
            text first.
        emotion:
            Emotion label used to adjust rate and volume.

        Returns
        -------
        SynthesisResult
            Synthesized audio bytes in WAV format.

        Raises
        ------
        intent_engine.errors.TTSError
            If there is no text to speak, or the engine produces no audio.
        """
        plain_text = strip_ssml(text)
        if not plain_text.strip():
            raise TTSError("eSpeak was given no text to synthesize.")

        # pyttsx3 blocks until eSpeak is done, so keep it off the event loop.
        audio_bytes = await asyncio.to_thread(
            self._synthesize_blocking, plain_text, emotion
        )

        return SynthesisResult(
            audio_data=audio_bytes,
            format="wav",
            sample_rate=22050,
        )
