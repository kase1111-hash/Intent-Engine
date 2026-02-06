"""Performance benchmark suite.

Measures latency of key pipeline operations to ensure they meet
performance targets from spec Section 9.  Uses lightweight mocks
so benchmarks run fast in CI.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from prosody_protocol import (
    ProsodyMapping,
    ProsodyProfile,
    SpanFeatures,
    ValidationResult,
)

from intent_engine.engine import IntentEngine
from tests.conftest import (
    create_mocked_engine,
    make_iml_document,
    make_interpretation_result,
    make_span_features,
    make_synthesis_result,
    make_transcription_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_fast_pipeline(engine):
    """Configure mocks that return instantly for latency measurement."""
    engine._stt.transcribe = AsyncMock(
        return_value=make_transcription_result(text="Benchmark text")
    )
    engine._analyzer.analyze = MagicMock(return_value=[make_span_features()])
    engine._analyzer.detect_pauses = MagicMock(return_value=[])
    engine._assembler.assemble = MagicMock(return_value=make_iml_document())
    engine._parser.to_iml_string = MagicMock(
        return_value="<iml><utterance>Benchmark text</utterance></iml>"
    )
    engine._validator.validate = MagicMock(return_value=ValidationResult(valid=True))
    engine._emotion_classifier.classify = MagicMock(return_value=("neutral", 0.6))
    engine._llm.interpret = AsyncMock(
        return_value=make_interpretation_result()
    )
    engine._tts.synthesize = AsyncMock(
        return_value=make_synthesis_result()
    )


def _measure_async(coro_factory, iterations=10):
    """Measure average wall-clock time of an async operation."""
    loop = asyncio.get_event_loop()
    # Warm up
    loop.run_until_complete(coro_factory())

    start = time.perf_counter()
    for _ in range(iterations):
        loop.run_until_complete(coro_factory())
    elapsed = time.perf_counter() - start
    return elapsed / iterations


def _measure_sync(func, iterations=10):
    """Measure average wall-clock time of a synchronous operation."""
    # Warm up
    func()

    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start
    return elapsed / iterations


# ---------------------------------------------------------------------------
# Pipeline latency benchmarks
# ---------------------------------------------------------------------------


class TestProcessVoiceInputPerformance:
    """Latency benchmark for the full STT pipeline."""

    def test_process_voice_input_latency(self) -> None:
        """process_voice_input should complete quickly with mocked providers."""
        engine = create_mocked_engine()
        _setup_fast_pipeline(engine)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake audio data " * 100)
            path = f.name

        try:
            avg_time = _measure_async(
                lambda: engine.process_voice_input(path, use_cache=False),
                iterations=20,
            )
            # With mocked providers, this should be well under 100ms
            assert avg_time < 0.1, f"process_voice_input avg latency: {avg_time:.4f}s"
        finally:
            Path(path).unlink()

    def test_process_voice_input_with_cache_is_faster(self) -> None:
        """Cache hit should be significantly faster than full pipeline."""
        engine = create_mocked_engine()
        _setup_fast_pipeline(engine)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake audio data")
            path = f.name

        try:
            loop = asyncio.get_event_loop()
            # Prime cache
            loop.run_until_complete(engine.process_voice_input(path))

            # Measure cache hit
            start = time.perf_counter()
            for _ in range(50):
                loop.run_until_complete(engine.process_voice_input(path))
            cache_time = (time.perf_counter() - start) / 50

            # Cache hits should be very fast
            assert cache_time < 0.01, f"Cache hit avg latency: {cache_time:.4f}s"
        finally:
            Path(path).unlink()


class TestGenerateResponsePerformance:
    """Latency benchmark for LLM response generation."""

    def test_generate_response_latency(self) -> None:
        engine = create_mocked_engine()
        _setup_fast_pipeline(engine)

        avg_time = _measure_async(
            lambda: engine.generate_response("<iml/>", tone="calm"),
            iterations=50,
        )
        assert avg_time < 0.05, f"generate_response avg latency: {avg_time:.4f}s"


class TestSynthesizeSpeechPerformance:
    """Latency benchmark for TTS synthesis."""

    def test_synthesize_speech_latency(self) -> None:
        engine = create_mocked_engine()
        _setup_fast_pipeline(engine)

        avg_time = _measure_async(
            lambda: engine.synthesize_speech("Hello", emotion="joyful"),
            iterations=50,
        )
        assert avg_time < 0.05, f"synthesize_speech avg latency: {avg_time:.4f}s"


class TestEvaluateIntentPerformance:
    """Latency benchmark for constitutional filter evaluation."""

    def test_evaluate_intent_no_filter(self) -> None:
        """Without filter, evaluate_intent is a simple Decision return."""
        engine = create_mocked_engine()
        features = [make_span_features()]

        avg_time = _measure_sync(
            lambda: engine.evaluate_intent("action", features),
            iterations=100,
        )
        assert avg_time < 0.001, f"evaluate_intent (no filter) avg: {avg_time:.6f}s"

    def test_evaluate_intent_with_filter(self) -> None:
        """With filter rules, evaluation should still be fast."""
        from intent_engine.constitutional.filter import ConstitutionalFilter
        from intent_engine.constitutional.rules import ConstitutionalRule, ProsodyCondition

        engine = create_mocked_engine()
        rules = [
            ConstitutionalRule(
                name=f"rule_{i}",
                triggers=[f"action_{i}"],
                forbidden_prosody=ProsodyCondition(emotion=["sarcastic"]),
            )
            for i in range(10)
        ]
        engine._filter = ConstitutionalFilter(rules)
        features = [make_span_features()]

        avg_time = _measure_sync(
            lambda: engine.evaluate_intent("action_5", features, emotion="neutral"),
            iterations=100,
        )
        assert avg_time < 0.01, f"evaluate_intent (10 rules) avg: {avg_time:.6f}s"


# ---------------------------------------------------------------------------
# Feature label derivation performance
# ---------------------------------------------------------------------------


class TestDeriveFeatureLabelsPerformance:
    """Feature label derivation should be O(n) in span count."""

    def test_single_span_fast(self) -> None:
        features = [make_span_features()]
        avg_time = _measure_sync(
            lambda: IntentEngine._derive_feature_labels(features),
            iterations=1000,
        )
        assert avg_time < 0.001, f"_derive_feature_labels (1 span) avg: {avg_time:.6f}s"

    def test_many_spans_still_fast(self) -> None:
        features = [
            SpanFeatures(
                start_ms=i * 100,
                end_ms=(i + 1) * 100,
                text=f"word_{i}",
                f0_mean=150.0 + (i % 50),
                intensity_mean=60.0 + (i % 20),
                speech_rate=4.0 + (i % 3),
            )
            for i in range(100)
        ]
        avg_time = _measure_sync(
            lambda: IntentEngine._derive_feature_labels(features),
            iterations=100,
        )
        assert avg_time < 0.01, f"_derive_feature_labels (100 spans) avg: {avg_time:.6f}s"


# ---------------------------------------------------------------------------
# Profile application performance
# ---------------------------------------------------------------------------


class TestProfilePerformance:
    """Profile application overhead should be minimal."""

    def test_profile_apply_fast(self) -> None:
        engine = create_mocked_engine()
        profile = ProsodyProfile(
            profile_version="1.0",
            user_id="perf-test",
            description="Performance test profile",
            mappings=[
                ProsodyMapping(
                    pattern={"f0_mean": "high"},
                    interpretation_emotion="joyful",
                    confidence_boost=0.2,
                ),
                ProsodyMapping(
                    pattern={"speech_rate": "slow"},
                    interpretation_emotion="calm",
                    confidence_boost=0.1,
                ),
                ProsodyMapping(
                    pattern={"intensity_mean": "loud"},
                    interpretation_emotion="angry",
                    confidence_boost=0.15,
                ),
            ],
        )

        labels = {"f0_mean": "high", "intensity_mean": "normal", "speech_rate": "normal"}

        avg_time = _measure_sync(
            lambda: engine._profile_applier.apply(profile, labels, "neutral", 0.5),
            iterations=1000,
        )
        assert avg_time < 0.001, f"profile apply avg: {avg_time:.6f}s"


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------


class TestThroughput:
    """Measure how many operations per second the pipeline can handle."""

    def test_generate_response_throughput(self) -> None:
        """At least 100 generate_response calls/sec with mocks."""
        engine = create_mocked_engine()
        _setup_fast_pipeline(engine)
        loop = asyncio.get_event_loop()

        count = 100
        start = time.perf_counter()
        for _ in range(count):
            loop.run_until_complete(engine.generate_response("<iml/>"))
        elapsed = time.perf_counter() - start

        throughput = count / elapsed
        assert throughput > 100, f"generate_response throughput: {throughput:.0f} ops/sec"

    def test_evaluate_intent_throughput(self) -> None:
        """At least 1000 evaluate_intent calls/sec."""
        engine = create_mocked_engine()
        features = [make_span_features()]

        count = 1000
        start = time.perf_counter()
        for _ in range(count):
            engine.evaluate_intent("action", features)
        elapsed = time.perf_counter() - start

        throughput = count / elapsed
        assert throughput > 1000, f"evaluate_intent throughput: {throughput:.0f} ops/sec"
