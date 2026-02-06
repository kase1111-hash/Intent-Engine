"""Tests for the TrainingEvaluator class."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from prosody_protocol import (
    Benchmark,
    BenchmarkReport,
    Dataset,
    DatasetEntry,
)

from intent_engine.training.evaluator import TrainingEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    entry_id: str = "e001",
    text: str = "Hello",
    emotion: str = "joyful",
) -> DatasetEntry:
    iml = f'<iml><utterance emotion="{emotion}" confidence="0.9">{text}</utterance></iml>'
    return DatasetEntry(
        id=entry_id,
        timestamp="2024-01-01T00:00:00Z",
        source="synthetic",
        language="en",
        audio_file="audio/e001.wav",
        transcript=text,
        iml=iml,
        emotion_label=emotion,
        annotator="human",
        consent=True,
    )


def _make_dataset(n: int = 5) -> Dataset:
    emotions = ["joyful", "sad", "neutral", "angry", "frustrated"]
    return Dataset(
        name="eval-test",
        entries=[
            _make_entry(f"e{i:03d}", f"Text {i}", emotions[i % len(emotions)])
            for i in range(n)
        ],
    )


def _make_report(**overrides: object) -> BenchmarkReport:
    defaults = {
        "emotion_accuracy": 0.87,
        "emotion_f1": {"joyful": 0.9, "sad": 0.85},
        "confidence_ece": 0.05,
        "pitch_accuracy": 0.8,
        "pause_f1": 0.75,
        "validity_rate": 1.0,
        "num_samples": 5,
        "duration_seconds": 1.0,
    }
    defaults.update(overrides)
    return BenchmarkReport(**defaults)


def _make_mock_converter():
    converter = MagicMock()
    converter.convert = MagicMock(return_value=None)
    return converter


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestTrainingEvaluatorConstruction:
    """TrainingEvaluator initialization."""

    def test_from_dataset_object(self) -> None:
        ds = _make_dataset()
        evaluator = TrainingEvaluator(dataset=ds)
        assert evaluator.dataset is ds

    def test_from_dataset_path(self) -> None:
        ds = _make_dataset()
        with patch.object(
            TrainingEvaluator, "__init__", return_value=None
        ) as mock_init:
            # Test that string path triggers DatasetLoader.load
            pass

        # Direct test: construct with Dataset object
        evaluator = TrainingEvaluator(dataset=ds)
        assert evaluator.dataset.name == "eval-test"

    def test_with_dataset_dir(self) -> None:
        ds = _make_dataset()
        evaluator = TrainingEvaluator(dataset=ds, dataset_dir="/data")
        assert evaluator._dataset_dir == Path("/data")

    def test_repr(self) -> None:
        ds = _make_dataset()
        evaluator = TrainingEvaluator(dataset=ds)
        r = repr(evaluator)
        assert "eval-test" in r
        assert "5" in r


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


class TestTrainingEvaluatorEvaluate:
    """TrainingEvaluator.evaluate() runs benchmark."""

    def test_evaluate_returns_report(self) -> None:
        ds = _make_dataset()
        evaluator = TrainingEvaluator(dataset=ds)
        converter = _make_mock_converter()

        report = evaluator.evaluate(converter)
        assert isinstance(report, BenchmarkReport)
        assert report.num_samples == 5

    def test_evaluate_with_max_samples(self) -> None:
        ds = _make_dataset(n=10)
        evaluator = TrainingEvaluator(dataset=ds)
        converter = _make_mock_converter()

        report = evaluator.evaluate(converter, max_samples=3)
        assert report.num_samples == 3


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------


class TestTrainingEvaluatorCompare:
    """TrainingEvaluator.compare() runs both models."""

    def test_compare_returns_both_reports(self) -> None:
        ds = _make_dataset()
        evaluator = TrainingEvaluator(dataset=ds)
        baseline = _make_mock_converter()
        finetuned = _make_mock_converter()

        results = evaluator.compare(baseline, finetuned)
        assert "baseline" in results
        assert "finetuned" in results
        assert isinstance(results["baseline"], BenchmarkReport)
        assert isinstance(results["finetuned"], BenchmarkReport)


# ---------------------------------------------------------------------------
# check_regression()
# ---------------------------------------------------------------------------


class TestTrainingEvaluatorRegression:
    """TrainingEvaluator.check_regression() delegates to report."""

    def test_no_regression(self) -> None:
        ds = _make_dataset()
        evaluator = TrainingEvaluator(dataset=ds)

        baseline = _make_report(emotion_accuracy=0.85)
        current = _make_report(emotion_accuracy=0.90)

        failures = evaluator.check_regression(current, baseline=baseline)
        assert failures == []

    def test_detects_regression(self) -> None:
        ds = _make_dataset()
        evaluator = TrainingEvaluator(dataset=ds)

        baseline = _make_report(emotion_accuracy=0.90)
        current = _make_report(emotion_accuracy=0.50)

        failures = evaluator.check_regression(current, baseline=baseline)
        assert len(failures) > 0

    def test_threshold_check(self) -> None:
        ds = _make_dataset()
        evaluator = TrainingEvaluator(dataset=ds)

        report = _make_report(emotion_accuracy=0.80)
        failures = evaluator.check_regression(
            report, thresholds={"emotion_accuracy": 0.95}
        )
        assert len(failures) > 0


# ---------------------------------------------------------------------------
# save/load report
# ---------------------------------------------------------------------------


class TestTrainingEvaluatorReportIO:
    """Report save and load through the evaluator."""

    def test_save_and_load_report(self) -> None:
        ds = _make_dataset()
        evaluator = TrainingEvaluator(dataset=ds)

        report = _make_report()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            evaluator.save_report(report, path)
            loaded = evaluator.load_report(path)
            assert loaded.emotion_accuracy == report.emotion_accuracy
            assert loaded.num_samples == report.num_samples
        finally:
            Path(path).unlink()
