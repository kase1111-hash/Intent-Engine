"""AssemblyAI STT adapter.

Uses the AssemblyAI SDK to transcribe audio via the AssemblyAI API.
Requires an ``ASSEMBLYAI_API_KEY`` environment variable or explicit
``api_key`` parameter.
"""

from __future__ import annotations

import logging
import os

from prosody_protocol import WordAlignment

from intent_engine.stt.base import STTProvider, TranscriptionResult

logger = logging.getLogger(__name__)


class AssemblyAISTT(STTProvider):
    """AssemblyAI API-based speech-to-text provider.

    Parameters
    ----------
    api_key:
        AssemblyAI API key.  Falls back to the ``ASSEMBLYAI_API_KEY``
        environment variable if not provided.
    language_code:
        Language code (e.g., ``"en"``).
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
        """
        try:
            import assemblyai as aai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "assemblyai is required for AssemblyAISTT. "
                "Install it with: pip install intent-engine[assemblyai]"
            ) from exc

        aai.settings.api_key = self._api_key

        config = aai.TranscriptionConfig(language_code=self._language_code)
        transcriber = aai.Transcriber(config=config)

        transcript = transcriber.transcribe(audio_path)

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")

        text: str = transcript.text or ""
        alignments: list[WordAlignment] = []

        for word_info in transcript.words or []:
            word_text: str = (word_info.text or "").strip()
            if not word_text:
                continue
            start_ms = word_info.start or 0
            end_ms = word_info.end or 0
            alignments.append(
                WordAlignment(word=word_text, start_ms=start_ms, end_ms=end_ms)
            )

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
