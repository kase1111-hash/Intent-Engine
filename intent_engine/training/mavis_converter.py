"""MavisDataConverter -- convert Mavis game data to training entries.

Wraps ``prosody_protocol.MavisBridge`` to convert Mavis vocal-typing
game sessions into ``DatasetEntry`` objects suitable for fine-tuning.
Provides batch conversion, feature extraction, and dataset export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from prosody_protocol import Dataset, DatasetEntry, MavisBridge
from prosody_protocol.mavis_bridge import PhonemeEvent

logger = logging.getLogger(__name__)


# Feature names matching MavisBridge.extract_training_features() output
FEATURE_NAMES = [
    "mean_pitch_hz",
    "pitch_range_hz",
    "mean_volume",
    "volume_range",
    "mean_breathiness",
    "speech_rate",
    "vibrato_ratio",
]


class MavisDataConverter:
    """Convert Mavis game data into Prosody Protocol training entries.

    Uses ``prosody_protocol.MavisBridge`` for the actual conversion
    logic.  This class adds batch operations, session management,
    and convenience methods for building training datasets.

    Parameters
    ----------
    language:
        BCP-47 language code for the generated entries.
    """

    def __init__(self, language: str = "en-US") -> None:
        self._bridge = MavisBridge(language=language)
        self._language = language

    @property
    def language(self) -> str:
        """The configured language code."""
        return self._language

    def convert_session(
        self,
        events: list[PhonemeEvent],
        transcript: str,
        session_id: str,
        emotion_label: str | None = None,
        speaker_id: str | None = None,
    ) -> DatasetEntry:
        """Convert a single Mavis session to a ``DatasetEntry``.

        Parameters
        ----------
        events:
            Phoneme events from a Mavis game session.
        transcript:
            The plain-text transcription of the session.
        session_id:
            Unique identifier for this session.
        emotion_label:
            Emotion label.  Auto-inferred by MavisBridge if ``None``.
        speaker_id:
            Optional speaker identifier.

        Returns
        -------
        DatasetEntry
            A validated dataset entry.
        """
        entry = self._bridge.phoneme_events_to_entry(
            events=events,
            transcript=transcript,
            session_id=session_id,
            emotion_label=emotion_label,
            speaker_id=speaker_id,
        )
        logger.debug("Converted session '%s' → entry '%s'", session_id, entry.id)
        return entry

    def convert_sessions(
        self,
        sessions: list[dict[str, Any]],
    ) -> list[DatasetEntry]:
        """Convert multiple Mavis sessions to dataset entries.

        Parameters
        ----------
        sessions:
            List of session dicts with keys:

            * ``"events"`` -- list of ``PhonemeEvent``
            * ``"transcript"`` -- str
            * ``"session_id"`` -- str
            * ``"emotion_label"`` -- optional str
            * ``"speaker_id"`` -- optional str

        Returns
        -------
        list[DatasetEntry]
            Converted entries.
        """
        entries: list[DatasetEntry] = []
        for session in sessions:
            entry = self.convert_session(
                events=session["events"],
                transcript=session["transcript"],
                session_id=session["session_id"],
                emotion_label=session.get("emotion_label"),
                speaker_id=session.get("speaker_id"),
            )
            entries.append(entry)

        logger.info("Converted %d Mavis sessions to dataset entries", len(entries))
        return entries

    def extract_features(
        self, events: list[PhonemeEvent]
    ) -> np.ndarray:
        """Extract a 7-dimensional feature vector from phoneme events.

        Uses ``MavisBridge.extract_training_features()``.

        Parameters
        ----------
        events:
            Phoneme events from a single session.

        Returns
        -------
        numpy.ndarray
            Shape ``(7,)`` with features:
            ``[mean_pitch_hz, pitch_range_hz, mean_volume,
            volume_range, mean_breathiness, speech_rate,
            vibrato_ratio]``.
        """
        return self._bridge.extract_training_features(events)

    def batch_extract_features(
        self, sessions: list[list[PhonemeEvent]]
    ) -> np.ndarray:
        """Extract features from multiple sessions.

        Parameters
        ----------
        sessions:
            List of phoneme event lists.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_sessions, 7)``.
        """
        return self._bridge.batch_extract_features(sessions)

    def export_dataset(
        self,
        sessions: list[dict[str, Any]],
        output_dir: str | Path,
    ) -> Dataset:
        """Export Mavis sessions as a Prosody Protocol dataset.

        Writes entry JSON files and metadata to ``output_dir``.

        Parameters
        ----------
        sessions:
            Session dicts (same format as :meth:`convert_sessions`).
        output_dir:
            Directory to write the dataset to.

        Returns
        -------
        Dataset
            The exported and loaded dataset.
        """
        dataset = self._bridge.export_dataset(sessions, output_dir)
        logger.info(
            "Exported dataset '%s' with %d entries to %s",
            dataset.name,
            dataset.size,
            output_dir,
        )
        return dataset

    @staticmethod
    def create_phoneme_event(
        phoneme: str,
        start_ms: int = 0,
        duration_ms: int = 100,
        volume: float = 0.5,
        pitch_hz: float = 220.0,
        vibrato: bool = False,
        breathiness: float = 0.0,
        harmony_intervals: list[int] | None = None,
    ) -> PhonemeEvent:
        """Create a ``PhonemeEvent`` with convenient defaults.

        Parameters
        ----------
        phoneme:
            IPA phoneme symbol (e.g., ``"a"``, ``"p"``, ``"ʃ"``).
        start_ms:
            Start time in milliseconds.
        duration_ms:
            Duration in milliseconds.
        volume:
            Volume level (0.0 -- 1.0).
        pitch_hz:
            Fundamental frequency in Hz.
        vibrato:
            Whether vibrato is present.
        breathiness:
            Breathiness level (0.0 -- 1.0).
        harmony_intervals:
            Optional harmony interval list.

        Returns
        -------
        PhonemeEvent
        """
        return PhonemeEvent(
            phoneme=phoneme,
            start_ms=start_ms,
            duration_ms=duration_ms,
            volume=volume,
            pitch_hz=pitch_hz,
            vibrato=vibrato,
            breathiness=breathiness,
            harmony_intervals=harmony_intervals,
        )

    def __repr__(self) -> str:
        return f"MavisDataConverter(language={self._language!r})"
