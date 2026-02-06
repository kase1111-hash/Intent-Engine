"""Abstract STT provider interface and shared types.

All STT adapters implement ``STTProvider`` and return
``TranscriptionResult`` objects containing word-level timestamps
as ``prosody_protocol.WordAlignment`` instances.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from prosody_protocol import WordAlignment


@dataclass(frozen=True)
class TranscriptionResult:
    """Output of an STT provider's ``transcribe()`` call.

    Attributes
    ----------
    text:
        Full transcription text.
    alignments:
        Per-word time boundaries from ``prosody_protocol.WordAlignment``.
        Each alignment has ``word``, ``start_ms``, and ``end_ms`` fields.
    language:
        Detected language code (e.g., ``"en"``), or ``None`` if unknown.
    """

    text: str
    alignments: list[WordAlignment]
    language: str | None = None


class STTProvider(ABC):
    """Abstract base class for speech-to-text provider adapters.

    Subclasses must implement :meth:`transcribe` which converts an audio
    file into text with word-level timestamps.  The orchestrator
    (``IntentEngine``) feeds these timestamps into
    ``prosody_protocol.ProsodyAnalyzer`` and ``prosody_protocol.IMLAssembler``
    to produce IML output.
    """

    @abstractmethod
    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an audio file to text with word-level timestamps.

        Parameters
        ----------
        audio_path:
            Path to the audio file (WAV, MP3, etc.).

        Returns
        -------
        TranscriptionResult
            Transcription text and ``WordAlignment`` list.

        Raises
        ------
        intent_engine.errors.STTError
            If transcription fails.
        """
        ...
