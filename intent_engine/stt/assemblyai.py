"""AssemblyAI STT adapter.

Uses the AssemblyAI SDK to transcribe audio via the AssemblyAI API.
Requires an ``ASSEMBLYAI_API_KEY`` environment variable or explicit
``api_key`` parameter.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any

from prosody_protocol import ConversionError, WordAlignment
from prosody_protocol.alignment import from_assemblyai

from intent_engine.errors import STTError
from intent_engine.stt.base import STTProvider, TranscriptionResult

logger = logging.getLogger(__name__)

# The SDK reads the API key from process-wide settings when a ``Transcriber``
# is built, so setting the key and building the transcriber must not interleave
# between adapters (with different keys) running in worker threads.
_SETTINGS_LOCK = threading.Lock()


class AssemblyAISTT(STTProvider):
    """AssemblyAI API-based speech-to-text provider.

    Parameters
    ----------
    api_key:
        AssemblyAI API key.  Falls back to the ``ASSEMBLYAI_API_KEY``
        environment variable if not provided.
    language_code:
        Language code (e.g., ``"en"``).

    Notes
    -----
    The SDK's synchronous client uploads the file and polls until the
    transcript is done, in a worker thread.  A thread cannot be interrupted:
    cancelling a call stops the wait, but the SDK call keeps running until
    AssemblyAI finishes the job.
    """

    def __init__(
        self,
        api_key: str | None = None,
        language_code: str = "en",
        **kwargs: object,
    ) -> None:
        self._api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "AssemblyAI API key is required. Set ASSEMBLYAI_API_KEY or pass api_key=."
            )
        self._language_code = language_code

    def _transcribe_blocking(self, aai: Any, audio_path: str) -> Any:
        """Upload and transcribe with the SDK's synchronous client (worker thread)."""
        with _SETTINGS_LOCK:
            aai.settings.api_key = self._api_key
            config = aai.TranscriptionConfig(language_code=self._language_code)
            transcriber = aai.Transcriber(config=config)
        return transcriber.transcribe(audio_path)

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio via the AssemblyAI API.

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
            If ``assemblyai`` is not installed.
        FileNotFoundError
            If *audio_path* does not exist.
        STTError
            If the request fails, the transcript failed, or its word
            timings are unusable.
        """
        try:
            import assemblyai as aai
        except ImportError as exc:
            raise ImportError(
                "assemblyai is required for AssemblyAISTT. "
                "Install it with: pip install intent-engine[assemblyai]"
            ) from exc

        # The SDK call uploads the file and polls until done; keep it off the event loop.
        try:
            transcript = await asyncio.to_thread(self._transcribe_blocking, aai, audio_path)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise STTError(f"AssemblyAI request failed: {type(exc).__name__}: {exc}") from exc

        if transcript.status == aai.TranscriptStatus.error:
            raise STTError(f"AssemblyAI transcription failed: {transcript.error}")

        text: str = transcript.text or ""
        alignments: list[WordAlignment] = []
        if transcript.words:
            try:
                alignments = from_assemblyai(transcript)
            except ConversionError as exc:
                raise STTError(f"AssemblyAI returned unusable word timings: {exc}") from exc

        logger.info(
            "AssemblyAI transcribed %d words (language=%s)",
            len(alignments),
            self._language_code,
        )

        return TranscriptionResult(
            text=text,
            alignments=alignments,
            language=self._language_code,
        )
