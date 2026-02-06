"""Deepgram STT adapter.

Uses the Deepgram SDK to transcribe audio via the Deepgram API.
Requires a ``DEEPGRAM_API_KEY`` environment variable or explicit
``api_key`` parameter.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from prosody_protocol import WordAlignment

from intent_engine.stt.base import STTProvider, TranscriptionResult

logger = logging.getLogger(__name__)


class DeepgramSTT(STTProvider):
    """Deepgram API-based speech-to-text provider.

    Parameters
    ----------
    api_key:
        Deepgram API key.  Falls back to the ``DEEPGRAM_API_KEY``
        environment variable if not provided.
    model:
        Deepgram model to use (e.g., ``"nova-2"``).
    language:
        Language code (e.g., ``"en"``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "nova-2",
        language: str = "en",
        **kwargs: object,
    ) -> None:
        self._api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Deepgram API key is required. Set DEEPGRAM_API_KEY or pass api_key=."
            )
        self._model = model
        self._language = language

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio via the Deepgram API.

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
            from deepgram import DeepgramClient, PrerecordedOptions  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "deepgram-sdk is required for DeepgramSTT. "
                "Install it with: pip install intent-engine[deepgram]"
            ) from exc

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        client = DeepgramClient(self._api_key)

        buffer_data = path.read_bytes()
        payload: dict[str, Any] = {"buffer": buffer_data}

        options = PrerecordedOptions(
            model=self._model,
            language=self._language,
            smart_format=True,
            utterances=True,
            punctuate=True,
        )

        response = client.listen.rest.v("1").transcribe_file(payload, options)

        text = ""
        alignments: list[WordAlignment] = []
        detected_language: str | None = None

        results = response.get("results", {})
        channels = results.get("channels", [])
        if channels:
            channel = channels[0]
            detected_language = channel.get("detected_language")
            alternatives = channel.get("alternatives", [])
            if alternatives:
                text = alternatives[0].get("transcript", "")
                for word_info in alternatives[0].get("words", []):
                    word_text: str = word_info.get("word", "").strip()
                    if not word_text:
                        continue
                    start_ms = int(word_info.get("start", 0) * 1000)
                    end_ms = int(word_info.get("end", 0) * 1000)
                    alignments.append(
                        WordAlignment(word=word_text, start_ms=start_ms, end_ms=end_ms)
                    )

        logger.info(
            "Deepgram transcribed %d words (model=%s, language=%s)",
            len(alignments),
            self._model,
            detected_language or self._language,
        )

        return TranscriptionResult(
            text=text,
            alignments=alignments,
            language=detected_language or self._language,
        )
