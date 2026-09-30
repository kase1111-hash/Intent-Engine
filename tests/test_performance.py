"""Performance smoke tests for the pipeline operations.

Uses lightweight mocks so they run fast in CI.  What each operation does is
asserted exactly, through the call counts of the mocked providers.  Wall-clock
bounds are only a smoke check against something pathological (a stray sleep, a
quadratic loop): they leave two orders of magnitude of headroom over the
measured cost, so CPU contention on a shared runner cannot fail them.  The
latency targets of spec Section 9 concern real providers, which mocks cannot
measure.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from prosody_protocol import (
    ProsodyMapping,
    ProsodyProfile,
    ValidationResult,
)

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
    engine._llm.interpret = AsyncMock(
        return_value=make_interpretation_result()
    )
    engine._tts.synthesize = AsyncMock(
        return_value=make_synthesis_result()
    )


def _measure_async(coro_factory, iterations=10):
    """Measure average wall-clock time of an async operation.

    The warm-up call and the timed calls run in one event loop, so the
    figure is the operation's own cost and not that of creating a loop and
    its thread pool for every call.  The operation runs ``iterations + 1``
    times.
    """

    async def run():
        await coro_factory()  # warm up

        start = time.perf_counter()
        for _ in range(iterations):
            await coro_factory()
        return (time.perf_counter() - start) / iterations

    return asyncio.run(run())


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
    """Cost of the full STT pipeline."""

    def test_process_voice_input_latency(self) -> None:
        """Without the cache every call runs the pipeline, and quickly."""
        engine = create_mocked_engine()
        _setup_fast_pipeline(engine)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake audio data " * 100)
            path = f.name

        try:
            iterations = 20
            avg_time = _measure_async(
                lambda: engine.process_voice_input(path, use_cache=False),
                iterations=iterations,
            )
            # use_cache=False: the warm-up call and each timed call ran STT
            assert engine._stt.transcribe.await_count == iterations + 1
            # With mocked providers this takes about a millisecond
            assert avg_time < 0.5, f"process_voice_input avg latency: {avg_time:.4f}s"
        finally:
            Path(path).unlink()

    def test_a_cache_hit_does_not_rerun_the_pipeline(self) -> None:
        """Repeated calls on one file are served from the cache.

        Which providers ran is exact; the time is only a smoke bound (a hit
        costs a fraction of a millisecond on a running loop).
        """
        engine = create_mocked_engine()
        _setup_fast_pipeline(engine)
        stages = [
            engine._stt.transcribe,
            engine._analyzer.analyze,
            engine._assembler.assemble,
            engine._validator.validate,
        ]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake audio data")
            path = f.name

        async def prime_then_hit(hits: int) -> float:
            await engine.process_voice_input(path)
            ran_once = [stage.call_count for stage in stages]
            assert ran_once[0] == 1, "the first call must run STT"

            start = time.perf_counter()
            for _ in range(hits):
                await engine.process_voice_input(path)
            elapsed = time.perf_counter() - start

            assert [stage.call_count for stage in stages] == ran_once, "a hit re-ran a stage"
            return elapsed / hits

        try:
            cache_time = asyncio.run(prime_then_hit(50))
            assert cache_time < 0.1, f"Cache hit avg latency: {cache_time:.4f}s"
        finally:
            Path(path).unlink()


class TestGenerateResponsePerformance:
    """Cost of LLM response generation."""

    def test_generate_response_latency(self) -> None:
        engine = create_mocked_engine()
        _setup_fast_pipeline(engine)

        iterations = 50
        avg_time = _measure_async(
            lambda: engine.generate_response("<iml/>", tone="calm"),
            iterations=iterations,
        )
        assert engine._llm.interpret.await_count == iterations + 1
        assert avg_time < 0.25, f"generate_response avg latency: {avg_time:.4f}s"


class TestSynthesizeSpeechPerformance:
    """Cost of TTS synthesis."""

    def test_synthesize_speech_latency(self) -> None:
        engine = create_mocked_engine()
        _setup_fast_pipeline(engine)

        iterations = 50
        avg_time = _measure_async(
            lambda: engine.synthesize_speech("Hello", emotion="joyful"),
            iterations=iterations,
        )
        assert engine._tts.synthesize.await_count == iterations + 1
        assert avg_time < 0.25, f"synthesize_speech avg latency: {avg_time:.4f}s"


class TestEvaluateIntentPerformance:
    """Cost of constitutional filter evaluation."""

    def test_evaluate_intent_no_filter(self) -> None:
        """Without filter, evaluate_intent is a simple Decision return."""
        engine = create_mocked_engine()
        features = [make_span_features()]

        avg_time = _measure_sync(
            lambda: engine.evaluate_intent("action", features),
            iterations=100,
        )
        assert engine.evaluate_intent("action", features).allow
        assert avg_time < 0.05, f"evaluate_intent (no filter) avg: {avg_time:.6f}s"

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
        assert avg_time < 0.05, f"evaluate_intent (10 rules) avg: {avg_time:.6f}s"


# ---------------------------------------------------------------------------
# Profile application performance
# ---------------------------------------------------------------------------


class TestProfilePerformance:
    """Switching the active profile should be cheap."""

    def test_set_profile_fast(self) -> None:
        engine = create_mocked_engine()
        profile = ProsodyProfile(
            profile_version="1.0.0",
            user_id="perf-test",
            description="Performance test profile",
            mappings=(
                ProsodyMapping(
                    pattern={"pitch": "high"},
                    interpretation_emotion="joyful",
                    confidence_boost=0.2,
                ),
                ProsodyMapping(
                    pattern={"rate": "slow"},
                    interpretation_emotion="calm",
                    confidence_boost=0.1,
                ),
                ProsodyMapping(
                    pattern={"volume": "loud"},
                    interpretation_emotion="angry",
                    confidence_boost=0.15,
                ),
            ),
        )

        avg_time = _measure_sync(
            lambda: engine.set_profile(profile),
            iterations=200,
        )
        assert avg_time < 0.05, f"set_profile avg: {avg_time:.6f}s"


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------


class TestThroughput:
    """Many calls in a row: every one completes, at a rate no sane build misses."""

    def test_generate_response_throughput(self) -> None:
        """Every call reaches the LLM once; at least 10 calls/sec with mocks."""
        engine = create_mocked_engine()
        _setup_fast_pipeline(engine)

        count = 100
        start = time.perf_counter()
        for _ in range(count):
            asyncio.run(engine.generate_response("<iml/>"))
        elapsed = time.perf_counter() - start

        assert engine._llm.interpret.await_count == count
        throughput = count / elapsed
        assert throughput > 10, f"generate_response throughput: {throughput:.0f} ops/sec"

    def test_evaluate_intent_throughput(self) -> None:
        """Every call is answered; at least 100 calls/sec."""
        engine = create_mocked_engine()
        features = [make_span_features()]

        count = 1000
        start = time.perf_counter()
        decisions = [engine.evaluate_intent("action", features) for _ in range(count)]
        elapsed = time.perf_counter() - start

        assert len(decisions) == count
        assert all(decision.allow for decision in decisions)
        throughput = count / elapsed
        assert throughput > 100, f"evaluate_intent throughput: {throughput:.0f} ops/sec"
