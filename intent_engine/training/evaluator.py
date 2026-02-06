"""TrainingEvaluator -- compare fine-tuned models vs baselines.

Uses ``prosody_protocol.Benchmark`` and ``prosody_protocol.BenchmarkReport``
to evaluate model accuracy on labeled datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

from prosody_protocol import (
    Benchmark,
    BenchmarkReport,
    Dataset,
    DatasetLoader,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Converter protocol (matches Benchmark's expectation)
# ---------------------------------------------------------------------------


class Converter(Protocol):
    """Protocol for objects that convert audio to IML strings."""

    def convert(self, audio_path: str | Path) -> str: ...


# ---------------------------------------------------------------------------
# TrainingEvaluator
# ---------------------------------------------------------------------------


class TrainingEvaluator:
    """Evaluate fine-tuned models against baselines.

    Wraps ``prosody_protocol.Benchmark`` to run accuracy evaluations
    and regression checks for trained Intent Engine models.

    Parameters
    ----------
    dataset:
        A loaded ``Dataset`` or path to a dataset directory.
    dataset_dir:
        Root directory of the dataset (for resolving audio paths).
        If ``None``, ground-truth IML is used instead of running
        audio through a converter.
    """

    def __init__(
        self,
        dataset: Dataset | str,
        dataset_dir: str | Path | None = None,
    ) -> None:
        self._loader = DatasetLoader(validate_iml=True)

        if isinstance(dataset, str):
            self._dataset = self._loader.load(dataset)
        else:
            self._dataset = dataset

        self._dataset_dir = Path(dataset_dir) if dataset_dir else None

    @property
    def dataset(self) -> Dataset:
        """The evaluation dataset."""
        return self._dataset

    def evaluate(
        self,
        converter: Converter,
        max_samples: int | None = None,
    ) -> BenchmarkReport:
        """Run a benchmark evaluation.

        Parameters
        ----------
        converter:
            An object implementing ``convert(audio_path) -> str``
            that produces IML from audio.
        max_samples:
            Limit evaluation to the first N entries (useful for
            quick CI checks).

        Returns
        -------
        BenchmarkReport
            Full evaluation report with accuracy metrics.
        """
        benchmark = Benchmark(
            dataset=self._dataset,
            converter=converter,
            dataset_dir=self._dataset_dir,
        )

        report = benchmark.run(max_samples=max_samples)

        logger.info(
            "Evaluation complete: emotion_accuracy=%.3f, validity_rate=%.3f, "
            "samples=%d, duration=%.1fs",
            report.emotion_accuracy,
            report.validity_rate,
            report.num_samples,
            report.duration_seconds,
        )

        return report

    def compare(
        self,
        baseline_converter: Converter,
        finetuned_converter: Converter,
        max_samples: int | None = None,
    ) -> dict[str, BenchmarkReport]:
        """Compare a baseline model against a fine-tuned model.

        Parameters
        ----------
        baseline_converter:
            Converter for the baseline (prompt-based) model.
        finetuned_converter:
            Converter for the fine-tuned model.
        max_samples:
            Limit per evaluation.

        Returns
        -------
        dict[str, BenchmarkReport]
            Keys ``"baseline"`` and ``"finetuned"`` with their
            respective reports.
        """
        logger.info("Evaluating baseline model...")
        baseline_report = self.evaluate(baseline_converter, max_samples)

        logger.info("Evaluating fine-tuned model...")
        finetuned_report = self.evaluate(finetuned_converter, max_samples)

        # Log improvement summary
        acc_diff = finetuned_report.emotion_accuracy - baseline_report.emotion_accuracy
        logger.info(
            "Comparison: baseline_acc=%.3f, finetuned_acc=%.3f, improvement=%+.3f",
            baseline_report.emotion_accuracy,
            finetuned_report.emotion_accuracy,
            acc_diff,
        )

        return {
            "baseline": baseline_report,
            "finetuned": finetuned_report,
        }

    def check_regression(
        self,
        report: BenchmarkReport,
        baseline: BenchmarkReport | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> list[str]:
        """Check for performance regressions.

        Delegates to ``BenchmarkReport.check_regression()``.

        Parameters
        ----------
        report:
            The report to check.
        baseline:
            Previous baseline to compare against.
        thresholds:
            Minimum acceptable values (e.g.,
            ``{"emotion_accuracy": 0.87}``).

        Returns
        -------
        list[str]
            List of failure messages.  Empty means all checks passed.
        """
        failures = report.check_regression(
            baseline=baseline,
            thresholds=thresholds,
        )
        if failures:
            logger.warning(
                "Regression check found %d issue(s): %s",
                len(failures),
                "; ".join(failures),
            )
        else:
            logger.info("Regression check passed")

        return failures

    def save_report(
        self, report: BenchmarkReport, path: str | Path
    ) -> None:
        """Save a benchmark report to disk."""
        report.save(path)
        logger.info("Report saved to %s", path)

    def load_report(self, path: str | Path) -> BenchmarkReport:
        """Load a previously saved benchmark report."""
        return BenchmarkReport.load(path)

    def __repr__(self) -> str:
        return (
            f"TrainingEvaluator(dataset={self._dataset.name!r}, "
            f"size={self._dataset.size})"
        )
