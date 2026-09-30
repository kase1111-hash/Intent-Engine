"""Whisper STT adapter.

Uses OpenAI's Whisper model (via the ``openai-whisper`` package) to
transcribe audio locally.
No API key required -- the model runs on the local machine.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

from prosody_protocol import ConversionError, WordAlignment
from prosody_protocol.alignment import from_whisper

from intent_engine.errors import STTError
from intent_engine.stt.base import STTProvider, TranscriptionResult

logger = logging.getLogger(__name__)


class WhisperSTT(STTProvider):
    """Local Whisper-based speech-to-text provider.

    Parameters
    ----------
    model_size:
        Whisper model size.  One of ``"tiny"``, ``"base"``, ``"small"``,
        ``"medium"``, ``"large"``, ``"large-v2"``, ``"large-v3"``.
        Defaults to ``"base"``.
    device:
        Device to run on (``"cpu"``, ``"cuda"``).  Defaults to ``"cpu"``.
    language:
        Optional language code (e.g., ``"en"``).  If ``None``, Whisper
        auto-detects the language.

    Notes
    -----
    Loading the model and decoding run in a worker thread, one call at a
    time per instance.  A thread cannot be interrupted: cancelling a call
    stops the wait, but the decode finishes, its result is discarded and
    calls queued behind it still run in turn.  Bound the calls in flight if
    you apply timeouts.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: str | None = None,
        **kwargs: object,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._language = language
        self._model: Any = None
        # Loading the model is slow and overlapping decodes on one model are not
        # safe, so worker threads take turns.
        self._lock = threading.Lock()

    def _load_model(self) -> Any:
        """Lazily load the Whisper model on first use."""
        if self._model is None:
            try:
                import whisper
            except ImportError as exc:
                raise ImportError(
                    "openai-whisper is required for WhisperSTT. "
                    "Install it with: pip install intent-engine[whisper]"
                ) from exc
            logger.info("Loading Whisper model '%s' on %s", self._model_size, self._device)
            self._model = whisper.load_model(self._model_size, device=self._device)
        return self._model

    def _transcribe_blocking(self, audio_path: str) -> dict[str, Any]:
        """Load the model if needed and run Whisper (blocking, CPU/GPU bound)."""
        with self._lock:
            model = self._load_model()

            import whisper

            result: dict[str, Any] = whisper.transcribe(
                model,
                audio_path,
                language=self._language,
                word_timestamps=True,
            )
            return result

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio using the local Whisper model.

        Parameters
        ----------
        audio_path:
            Path to the audio file.

        Returns
        -------
        TranscriptionResult
            Transcription with word-level ``WordAlignment`` timestamps.

        Raises
        ------
        ImportError
            If ``openai-whisper`` is not installed.
        FileNotFoundError
            If *audio_path* does not exist.
        STTError
            If Whisper fails or its word timestamps are unusable.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Model loading and decoding take seconds to minutes; keep them off the event loop.
        try:
            result = await asyncio.to_thread(self._transcribe_blocking, str(path))
        except ImportError:
            raise
        except Exception as exc:
            raise STTError(f"Whisper transcription failed: {type(exc).__name__}: {exc}") from exc

        text: str = result.get("text", "").strip()
        detected_language: str | None = result.get("language")

        try:
            alignments: list[WordAlignment] = from_whisper(result)
        except ConversionError as exc:
            raise STTError(f"Whisper returned unusable word timings: {exc}") from exc

        logger.info(
            "Whisper transcribed %d words from %s (language=%s)",
            len(alignments),
            path.name,
            detected_language,
        )

        return TranscriptionResult(
            text=text,
            alignments=alignments,
            language=detected_language,
        )
