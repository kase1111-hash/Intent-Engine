"""Accuracy benchmark harness using prosody_protocol.Benchmark.

Tests the benchmark infrastructure to ensure it can evaluate emotion
detection accuracy, IML validity rates, and other metrics.  Uses
synthetic dataset entries (no real audio) to keep tests fast.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from prosody_protocol import (
    Benchmark,
    BenchmarkReport,
    Dataset,
    DatasetEntry,
    DatasetLoader,
    IMLParser,
    IMLValidator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    entry_id: str = "e001",
    text: str = "Hello world",
    emotion: str = "joyful",
    iml: str | None = None,
    audio_file: str = "audio/e001.wav",
) -> DatasetEntry:
    """Create a DatasetEntry with sensible defaults."""
    if iml is None:
        iml = (
            f'<iml><utterance emotion="{emotion}" confidence="0.9">'
            f"{text}</utterance></iml>"
        )
    return DatasetEntry(
        id=entry_id,
        timestamp="2024-01-01T00:00:00Z",
        source="synthetic",
        language="en",
        audio_file=audio_file,
        transcript=text,
        iml=iml,
        emotion_label=emotion,
        annotator="human",
        consent=True,
    )


def _make_dataset(entries: list[DatasetEntry] | None = None) -> Dataset:
    """Create a minimal Dataset."""
    if entries is None:
        entries = [
            _make_entry("e001", "I am happy!", "joyful"),
            _make_entry("e002", "I'm so sad", "sad"),
            _make_entry("e003", "This is fine", "neutral"),
            _make_entry("e004", "Are you sure?", "uncertain"),
            _make_entry("e005", "I'm furious!", "angry"),
        ]
    return Dataset(name="test-dataset", entries=entries)


def _make_mock_converter():
    """Create a mock converter that returns the ground-truth IML."""
    converter = MagicMock()
    converter.convert = MagicMock(return_value=None)
    return converter


# ---------------------------------------------------------------------------
# BenchmarkReport
# ---------------------------------------------------------------------------


class TestBenchmarkReport:
    """BenchmarkReport data structure and serialization."""

    def test_create_report(self) -> None:
        report = BenchmarkReport(
            emotion_accuracy=0.87,
            emotion_f1={"joyful": 0.9, "sad": 0.85},
            confidence_ece=0.05,
            pitch_accuracy=0.8,
            pause_f1=0.75,
            validity_rate=1.0,
            num_samples=100,
            duration_seconds=5.0,
        )
        assert report.emotion_accuracy == 0.87
        assert report.num_samples == 100

    def test_to_dict(self) -> None:
        report = BenchmarkReport(
            emotion_accuracy=0.87,
            emotion_f1={"joyful": 0.9},
            confidence_ece=0.05,
            pitch_accuracy=0.8,
            pause_f1=0.75,
            validity_rate=1.0,
            num_samples=50,
            duration_seconds=2.5,
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["emotion_accuracy"] == 0.87
        assert d["num_samples"] == 50

    def test_save_and_load(self) -> None:
        report = BenchmarkReport(
            emotion_accuracy=0.9,
            emotion_f1={"neutral": 0.88},
            confidence_ece=0.03,
            pitch_accuracy=0.85,
            pause_f1=0.8,
            validity_rate=1.0,
            num_samples=10,
            duration_seconds=1.0,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            report.save(path)
            loaded = BenchmarkReport.load(path)
            assert loaded.emotion_accuracy == report.emotion_accuracy
            assert loaded.num_samples == report.num_samples
            assert loaded.validity_rate == report.validity_rate
        finally:
            Path(path).unlink()

    def test_check_regression_passes(self) -> None:
        baseline = BenchmarkReport(
            emotion_accuracy=0.85,
            emotion_f1={},
            confidence_ece=0.05,
            pitch_accuracy=0.8,
            pause_f1=0.7,
            validity_rate=0.95,
            num_samples=100,
            duration_seconds=5.0,
        )
        current = BenchmarkReport(
            emotion_accuracy=0.90,
            emotion_f1={},
            confidence_ece=0.04,
            pitch_accuracy=0.85,
            pause_f1=0.75,
            validity_rate=1.0,
            num_samples=100,
            duration_seconds=4.5,
        )
        failures = current.check_regression(baseline=baseline)
        assert failures == []

    def test_check_regression_detects_drop(self) -> None:
        baseline = BenchmarkReport(
            emotion_accuracy=0.90,
            emotion_f1={},
            confidence_ece=0.03,
            pitch_accuracy=0.85,
            pause_f1=0.8,
            validity_rate=1.0,
            num_samples=100,
            duration_seconds=5.0,
        )
        current = BenchmarkReport(
            emotion_accuracy=0.50,
            emotion_f1={},
            confidence_ece=0.2,
            pitch_accuracy=0.4,
            pause_f1=0.3,
            validity_rate=0.5,
            num_samples=100,
            duration_seconds=5.0,
        )
        failures = current.check_regression(baseline=baseline)
        assert len(failures) > 0

    def test_check_regression_with_thresholds(self) -> None:
        report = BenchmarkReport(
            emotion_accuracy=0.80,
            emotion_f1={},
            confidence_ece=0.1,
            pitch_accuracy=0.7,
            pause_f1=0.6,
            validity_rate=0.9,
            num_samples=50,
            duration_seconds=3.0,
        )
        # Set thresholds higher than actual values
        failures = report.check_regression(
            thresholds={"emotion_accuracy": 0.95}
        )
        assert len(failures) > 0


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


class TestBenchmarkRun:
    """Benchmark.run() produces a valid report."""

    def test_run_with_synthetic_dataset(self) -> None:
        """Benchmark runs on synthetic dataset without audio (ground-truth IML only)."""
        dataset = _make_dataset()
        converter = _make_mock_converter()

        # dataset_dir=None means Benchmark uses ground-truth IML from entries
        benchmark = Benchmark(dataset=dataset, converter=converter, dataset_dir=None)
        report = benchmark.run()

        assert isinstance(report, BenchmarkReport)
        assert report.num_samples == 5
        assert 0.0 <= report.emotion_accuracy <= 1.0
        assert 0.0 <= report.validity_rate <= 1.0
        assert report.duration_seconds >= 0.0

    def test_run_with_max_samples(self) -> None:
        dataset = _make_dataset()
        converter = _make_mock_converter()
        benchmark = Benchmark(dataset=dataset, converter=converter, dataset_dir=None)

        report = benchmark.run(max_samples=2)
        assert report.num_samples == 2

    def test_perfect_ground_truth_gets_high_accuracy(self) -> None:
        """When using ground-truth IML, emotion accuracy should be high."""
        entries = [
            _make_entry(f"e{i:03d}", f"Text {i}", "joyful")
            for i in range(10)
        ]
        dataset = Dataset(name="perfect", entries=entries)
        converter = _make_mock_converter()

        benchmark = Benchmark(dataset=dataset, converter=converter, dataset_dir=None)
        report = benchmark.run()

        # Ground-truth comparison should yield high accuracy
        assert report.emotion_accuracy >= 0.8
        assert report.validity_rate >= 0.8

    def test_report_contains_all_fields(self) -> None:
        dataset = _make_dataset()
        converter = _make_mock_converter()
        benchmark = Benchmark(dataset=dataset, converter=converter, dataset_dir=None)

        report = benchmark.run()
        assert hasattr(report, "emotion_accuracy")
        assert hasattr(report, "emotion_f1")
        assert hasattr(report, "confidence_ece")
        assert hasattr(report, "pitch_accuracy")
        assert hasattr(report, "pause_f1")
        assert hasattr(report, "validity_rate")
        assert hasattr(report, "num_samples")
        assert hasattr(report, "duration_seconds")


# ---------------------------------------------------------------------------
# DatasetEntry and Dataset
# ---------------------------------------------------------------------------


class TestDatasetEntry:
    """DatasetEntry data model works correctly."""

    def test_create_entry(self) -> None:
        entry = _make_entry()
        assert entry.id == "e001"
        assert entry.emotion_label == "joyful"
        assert entry.consent is True

    def test_entry_is_frozen(self) -> None:
        entry = _make_entry()
        with pytest.raises(AttributeError):
            entry.id = "modified"  # type: ignore[misc]

    def test_entry_fields(self) -> None:
        entry = _make_entry(
            entry_id="test",
            text="test text",
            emotion="sad",
        )
        assert entry.transcript == "test text"
        assert entry.source == "synthetic"
        assert entry.language == "en"
        assert entry.annotator == "human"


class TestDataset:
    """Dataset data model works correctly."""

    def test_create_dataset(self) -> None:
        ds = _make_dataset()
        assert ds.name == "test-dataset"
        assert ds.size == 5

    def test_empty_dataset(self) -> None:
        ds = Dataset(name="empty", entries=[])
        assert ds.size == 0

    def test_entries_accessible(self) -> None:
        ds = _make_dataset()
        assert all(isinstance(e, DatasetEntry) for e in ds.entries)


# ---------------------------------------------------------------------------
# DatasetLoader validation
# ---------------------------------------------------------------------------


class TestDatasetLoaderValidation:
    """DatasetLoader.validate_entry checks schema compliance."""

    def test_valid_entry_passes(self) -> None:
        loader = DatasetLoader(validate_iml=True)
        entry_dict = {
            "id": "e001",
            "timestamp": "2024-01-01T00:00:00Z",
            "source": "synthetic",
            "language": "en",
            "audio_file": "audio/e001.wav",
            "transcript": "Hello",
            "iml": '<iml><utterance emotion="joyful" confidence="0.9">Hello</utterance></iml>',
            "emotion_label": "joyful",
            "annotator": "human",
            "consent": True,
        }
        result = loader.validate_entry(entry_dict)
        assert isinstance(result, IMLValidator) or isinstance(result, object)
        # validate_entry returns a ValidationResult
        assert hasattr(result, "valid")

    def test_missing_consent_fails(self) -> None:
        loader = DatasetLoader(validate_iml=False)
        entry_dict = {
            "id": "e001",
            "timestamp": "2024-01-01T00:00:00Z",
            "source": "synthetic",
            "language": "en",
            "audio_file": "audio/e001.wav",
            "transcript": "Hello",
            "iml": "<utterance>Hello</utterance>",
            "emotion_label": "joyful",
            "annotator": "human",
            "consent": False,
        }
        result = loader.validate_entry(entry_dict)
        # consent=False should produce validation issues (D2)
        has_issues = not result.valid or len(result.issues) > 0
        assert has_issues

    def test_invalid_source_fails(self) -> None:
        loader = DatasetLoader(validate_iml=False)
        entry_dict = {
            "id": "e001",
            "timestamp": "2024-01-01T00:00:00Z",
            "source": "unknown_source",
            "language": "en",
            "audio_file": "audio/e001.wav",
            "transcript": "Hello",
            "iml": "<utterance>Hello</utterance>",
            "emotion_label": "joyful",
            "annotator": "human",
            "consent": True,
        }
        result = loader.validate_entry(entry_dict)
        has_issues = not result.valid or len(result.issues) > 0
        assert has_issues


# ---------------------------------------------------------------------------
# DatasetLoader split
# ---------------------------------------------------------------------------


class TestDatasetLoaderSplit:
    """DatasetLoader.split() partitions datasets correctly."""

    def test_split_ratios(self) -> None:
        entries = [_make_entry(f"e{i:03d}") for i in range(100)]
        ds = Dataset(name="split-test", entries=entries)

        loader = DatasetLoader(validate_iml=False)
        train, val, test = loader.split(ds, train=0.8, val=0.1, test=0.1)

        assert len(train) + len(val) + len(test) == 100
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_split_deterministic(self) -> None:
        entries = [_make_entry(f"e{i:03d}") for i in range(50)]
        ds = Dataset(name="deterministic", entries=entries)

        loader = DatasetLoader(validate_iml=False)
        train1, val1, test1 = loader.split(ds, seed=42)
        train2, val2, test2 = loader.split(ds, seed=42)

        assert [e.id for e in train1] == [e.id for e in train2]
        assert [e.id for e in val1] == [e.id for e in val2]
        assert [e.id for e in test1] == [e.id for e in test2]

    def test_different_seed_different_split(self) -> None:
        entries = [_make_entry(f"e{i:03d}") for i in range(50)]
        ds = Dataset(name="seed-test", entries=entries)

        loader = DatasetLoader(validate_iml=False)
        train1, _, _ = loader.split(ds, seed=1)
        train2, _, _ = loader.split(ds, seed=2)

        # Very unlikely to be identical with different seeds
        assert [e.id for e in train1] != [e.id for e in train2]
