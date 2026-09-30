"""The tone hint given to the LLM never claims a reading the engine does not have.

``Result.suggested_tone`` is ``"neutral"`` whenever the engine abstained (no
emotion was reported), and the classifier never reports a measured neutral.
Passing it on as ``The user's voice sounds 'neutral'`` contradicted the IML
the model reads, whose prompt says not to assume the speaker is neutral.  A
``"neutral"`` tone therefore adds no hint; a real tone is still passed on.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from intent_engine.engine import IntentEngine
from intent_engine.stt.base import TranscriptionResult
from tests.conftest import create_mocked_engine, make_flat_speech, make_interpretation_result


def _engine() -> IntentEngine:
    engine = create_mocked_engine()
    engine._llm.interpret = AsyncMock(return_value=make_interpretation_result())
    return engine


def _context_sent(engine: IntentEngine) -> str | None:
    context: str | None = engine._llm.interpret.call_args.kwargs["context"]
    return context


class TestNeutralMeansNoReading:
    @pytest.mark.parametrize("tone", ["neutral", "Neutral", " neutral ", "NEUTRAL"])
    def test_a_neutral_tone_adds_no_hint(self, tone: str) -> None:
        engine = _engine()

        asyncio.run(engine.generate_response("<iml/>", tone=tone))

        assert _context_sent(engine) is None

    @pytest.mark.parametrize("tone", ["neutral", " Neutral"])
    def test_the_callers_own_context_is_passed_on_unchanged(self, tone: str) -> None:
        engine = _engine()

        asyncio.run(engine.generate_response("<iml/>", context="customer_support", tone=tone))

        assert _context_sent(engine) == "customer_support"

    @pytest.mark.parametrize("tone", [None, "", "   "])
    def test_no_tone_means_no_hint(self, tone: str | None) -> None:
        engine = _engine()

        asyncio.run(engine.generate_response("<iml/>", tone=tone))

        assert _context_sent(engine) is None


class TestExplicitTonesAreHonoured:
    @pytest.mark.parametrize("tone", ["angry", "sad", "joyful", "calm", "empathetic"])
    def test_a_real_tone_is_passed_on(self, tone: str) -> None:
        engine = _engine()

        asyncio.run(engine.generate_response("<iml/>", tone=tone))

        context = _context_sent(engine)
        assert context is not None
        assert f"'{tone}'" in context
        assert "sounds" in context

    def test_a_real_tone_follows_the_callers_context(self) -> None:
        engine = _engine()

        asyncio.run(engine.generate_response("<iml/>", context="customer_support", tone="angry"))

        context = _context_sent(engine)
        assert context is not None
        assert context.startswith("customer_support\n")
        assert "'angry'" in context

    def test_a_tone_that_only_resembles_neutral_is_still_passed_on(self) -> None:
        engine = _engine()

        asyncio.run(engine.generate_response("<iml/>", tone="neutral-ish"))

        assert _context_sent(engine) is not None


class TestAbstainedTurn:
    def test_the_documented_flow_sends_no_claim_when_no_emotion_was_reported(
        self, tmp_path: Path
    ) -> None:
        # generate_response(result.iml, tone=result.suggested_tone), as in the README
        engine = _engine()
        alignments, features = make_flat_speech()
        engine._stt.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text="I am fine thank you today.", alignments=alignments, language="en"
            )
        )
        engine._analyzer.analyze = MagicMock(return_value=features)
        engine._analyzer.detect_pauses = MagicMock(return_value=[])
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"RIFF fake audio")

        result = asyncio.run(engine.process_voice_input(str(audio)))
        assert (result.emotion, result.confidence) == ("neutral", 0.0)
        assert result.suggested_tone == "neutral"
        asyncio.run(engine.generate_response(result.iml, tone=result.suggested_tone))

        assert _context_sent(engine) is None
