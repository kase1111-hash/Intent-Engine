"""Tests for the MavisDataConverter class."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from prosody_protocol import Dataset, DatasetEntry
from prosody_protocol.mavis_bridge import PhonemeEvent

from intent_engine.training.mavis_converter import (
    FEATURE_NAMES,
    MavisDataConverter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_events(n: int = 5) -> list[PhonemeEvent]:
    """Create a list of simple phoneme events."""
    phonemes = ["h", "ɛ", "l", "oʊ", "z"]
    return [
        PhonemeEvent(
            phoneme=phonemes[i % len(phonemes)],
            start_ms=i * 100,
            duration_ms=100,
            volume=0.5 + (i * 0.05),
            pitch_hz=200.0 + (i * 10),
            vibrato=(i % 3 == 0),
            breathiness=0.1 * i,
        )
        for i in range(n)
    ]


def _make_session(
    session_id: str = "s001",
    transcript: str = "hello",
    emotion: str | None = None,
    n_events: int = 5,
) -> dict[str, Any]:
    return {
        "events": _make_events(n_events),
        "transcript": transcript,
        "session_id": session_id,
        "emotion_label": emotion,
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestMavisDataConverterConstruction:
    """MavisDataConverter initialization."""

    def test_default_language(self) -> None:
        converter = MavisDataConverter()
        assert converter.language == "en-US"

    def test_custom_language(self) -> None:
        converter = MavisDataConverter(language="ja-JP")
        assert converter.language == "ja-JP"

    def test_repr(self) -> None:
        converter = MavisDataConverter(language="fr-FR")
        assert "fr-FR" in repr(converter)


# ---------------------------------------------------------------------------
# PhonemeEvent creation
# ---------------------------------------------------------------------------


class TestCreatePhonemeEvent:
    """MavisDataConverter.create_phoneme_event() helper."""

    def test_create_with_defaults(self) -> None:
        event = MavisDataConverter.create_phoneme_event("a")
        assert event.phoneme == "a"
        assert event.start_ms == 0
        assert event.duration_ms == 100
        assert event.volume == 0.5
        assert event.pitch_hz == 220.0
        assert event.vibrato is False
        assert event.breathiness == 0.0

    def test_create_with_custom_values(self) -> None:
        event = MavisDataConverter.create_phoneme_event(
            "ʃ",
            start_ms=500,
            duration_ms=200,
            volume=0.8,
            pitch_hz=300.0,
            vibrato=True,
            breathiness=0.3,
            harmony_intervals=[3, 5],
        )
        assert event.phoneme == "ʃ"
        assert event.start_ms == 500
        assert event.vibrato is True
        assert event.harmony_intervals == [3, 5]


# ---------------------------------------------------------------------------
# Single session conversion
# ---------------------------------------------------------------------------


class TestConvertSession:
    """MavisDataConverter.convert_session() produces DatasetEntry."""

    def test_basic_conversion(self) -> None:
        converter = MavisDataConverter()
        events = _make_events()
        entry = converter.convert_session(
            events=events,
            transcript="hello",
            session_id="s001",
        )
        assert isinstance(entry, DatasetEntry)
        assert entry.transcript == "hello"
        assert entry.consent is True
        assert entry.source == "mavis"

    def test_with_emotion_label(self) -> None:
        converter = MavisDataConverter()
        events = _make_events()
        entry = converter.convert_session(
            events=events,
            transcript="I'm happy!",
            session_id="s002",
            emotion_label="joyful",
        )
        assert entry.emotion_label == "joyful"

    def test_with_speaker_id(self) -> None:
        converter = MavisDataConverter()
        events = _make_events()
        entry = converter.convert_session(
            events=events,
            transcript="test",
            session_id="s003",
            speaker_id="speaker_42",
        )
        assert entry.speaker_id == "speaker_42"

    def test_auto_inferred_emotion(self) -> None:
        converter = MavisDataConverter()
        events = _make_events()
        entry = converter.convert_session(
            events=events,
            transcript="test",
            session_id="s004",
        )
        # MavisBridge should auto-infer an emotion
        assert entry.emotion_label is not None
        assert len(entry.emotion_label) > 0

    def test_entry_has_iml(self) -> None:
        converter = MavisDataConverter()
        events = _make_events()
        entry = converter.convert_session(
            events=events,
            transcript="hello",
            session_id="s005",
        )
        assert entry.iml is not None
        assert len(entry.iml) > 0


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------


class TestConvertSessions:
    """MavisDataConverter.convert_sessions() batch operation."""

    def test_batch_conversion(self) -> None:
        converter = MavisDataConverter()
        sessions = [
            _make_session("s001", "hello"),
            _make_session("s002", "goodbye"),
            _make_session("s003", "thank you"),
        ]
        entries = converter.convert_sessions(sessions)
        assert len(entries) == 3
        assert all(isinstance(e, DatasetEntry) for e in entries)

    def test_empty_batch(self) -> None:
        converter = MavisDataConverter()
        entries = converter.convert_sessions([])
        assert entries == []


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    """Feature extraction from phoneme events."""

    def test_extract_single_session(self) -> None:
        converter = MavisDataConverter()
        events = _make_events()
        features = converter.extract_features(events)
        assert isinstance(features, np.ndarray)
        assert features.shape == (7,)

    def test_batch_extract(self) -> None:
        converter = MavisDataConverter()
        sessions = [_make_events(5), _make_events(3), _make_events(8)]
        features = converter.batch_extract_features(sessions)
        assert isinstance(features, np.ndarray)
        assert features.shape == (3, 7)

    def test_feature_names_match_count(self) -> None:
        assert len(FEATURE_NAMES) == 7

    def test_features_are_numeric(self) -> None:
        converter = MavisDataConverter()
        events = _make_events()
        features = converter.extract_features(events)
        assert features.dtype in (np.float32, np.float64)


# ---------------------------------------------------------------------------
# Dataset export
# ---------------------------------------------------------------------------


class TestExportDataset:
    """MavisDataConverter.export_dataset() writes to disk."""

    def test_export_creates_dataset(self) -> None:
        converter = MavisDataConverter()
        sessions = [
            _make_session("s001", "hello"),
            _make_session("s002", "world"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_export"
            dataset = converter.export_dataset(sessions, str(output_dir))
            assert isinstance(dataset, Dataset)
            assert dataset.size == 2
            assert output_dir.exists()
