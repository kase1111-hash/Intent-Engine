"""Whisper STT adapter.

Uses OpenAI's Whisper model (via the ``openai-whisper`` or
``faster-whisper`` package) to transcribe audio locally.
No API key required -- the model runs on the local machine.
"""

from __future__ import annotations

import logging
from pathlib import Path

from prosody_protocol import WordAlignment

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
        self._model: object | None = None

    def _load_model(self) -> object:
        """Lazily load the Whisper model on first use."""
        if self._model is None:
            try:
                import whisper  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "openai-whisper is required for WhisperSTT. "
                    "Install it with: pip install intent-engine[whisper]"
                ) from exc
            logger.info("Loading Whisper model '%s' on %s", self._model_size, self._device)
            self._model = whisper.load_model(self._model_size, device=self._device)
        return self._model

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
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self._load_model()

        import whisper  # type: ignore[import-untyped]

        result = whisper.transcribe(
            model,
            str(path),
            language=self._language,
            word_timestamps=True,
        )

        text: str = result.get("text", "").strip()
        detected_language: str | None = result.get("language")

        alignments: list[WordAlignment] = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                word_text: str = word_info.get("word", "").strip()
                if not word_text:
                    continue
                start_ms = int(word_info.get("start", 0) * 1000)
                end_ms = int(word_info.get("end", 0) * 1000)
                alignments.append(
                    WordAlignment(word=word_text, start_ms=start_ms, end_ms=end_ms)
                )

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
