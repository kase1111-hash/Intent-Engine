"""Integration tests: module interactions across the pipeline.

Tests that multiple Intent Engine modules work together correctly
when wired through mocked providers.  Each test exercises at least
two modules (e.g., STT + ProsodyAnalyzer, Engine + ConstitutionalFilter).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from prosody_protocol import (
    ProsodyMapping,
    ProsodyProfile,
    ValidationResult,
)

from intent_engine.constitutional.filter import ConstitutionalFilter
from intent_engine.constitutional.rules import ConstitutionalRule, ProsodyCondition
from intent_engine.models.audio import Audio
from intent_engine.models.response import Response
from intent_engine.stt.base import TranscriptionResult
from tests.conftest import (
    assert_valid_iml,
    create_mocked_engine,
    make_flat_speech,
    make_iml_document,
    make_interpretation_result,
    make_span_features,
    make_synthesis_result,
    make_transcription_result,
)


class TestSTTProsodyIntegration:
    """STT result flows correctly into prosody analysis + IML assembly."""

    def test_transcription_feeds_assembler(self) -> None:
        engine = create_mocked_engine()

        transcription = make_transcription_result(text="I'm really upset")
        engine._stt.transcribe = AsyncMock(return_value=transcription)
        engine._analyzer.analyze = MagicMock(return_value=[make_span_features()])
        engine._analyzer.detect_pauses = MagicMock(return_value=[])

        iml_doc = make_iml_document(emotion="angry", confidence=0.8)
        engine._assembler.assemble = MagicMock(return_value=iml_doc)
        engine._parser.to_iml_string = MagicMock(return_value="<iml/>")
        engine._validator.validate = MagicMock(return_value=ValidationResult(valid=True))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake")
            path = f.name

        try:
            result = asyncio.run(
                engine.process_voice_input(path)
            )
            assert result.text == "I'm really upset"
            assert result.emotion == "angry"
            assert result.confidence == 0.8
            engine._assembler.assemble.assert_called_once()
        finally:
            Path(path).unlink()

    def test_prosody_fallback_still_produces_result(self) -> None:
        engine = create_mocked_engine()

        engine._stt.transcribe = AsyncMock(
            return_value=make_transcription_result(text="Hello")
        )
        engine._analyzer.analyze = MagicMock(side_effect=RuntimeError("parselmouth error"))
        engine._analyzer.detect_pauses = MagicMock(side_effect=RuntimeError("error"))

        iml_doc = make_iml_document()
        engine._assembler.assemble = MagicMock(return_value=iml_doc)
        engine._parser.to_iml_string = MagicMock(return_value="<iml/>")
        engine._validator.validate = MagicMock(return_value=ValidationResult(valid=True))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake")
            path = f.name

        try:
            result = asyncio.run(
                engine.process_voice_input(path)
            )
            assert result.text == "Hello"
            assert result.emotion == "neutral"
        finally:
            Path(path).unlink()


class TestLLMResponseIntegration:
    """LLM interpretation wires into Response model correctly."""

    def test_iml_to_response(self) -> None:
        engine = create_mocked_engine()
        engine._llm.interpret = AsyncMock(
            return_value=make_interpretation_result(
                intent="cancel", response_text="I'll cancel that for you.",
                suggested_emotion="empathetic",
            )
        )

        response = asyncio.run(
            engine.generate_response("<utterance>Cancel my order</utterance>")
        )
        assert isinstance(response, Response)
        assert response.text == "I'll cancel that for you."
        assert response.emotion == "empathetic"

    def test_tone_hint_appended(self) -> None:
        engine = create_mocked_engine()
        engine._llm.interpret = AsyncMock(
            return_value=make_interpretation_result()
        )

        asyncio.run(
            engine.generate_response("<iml/>", context="Support", tone="calm")
        )

        call_args = engine._llm.interpret.call_args
        context_str = call_args.kwargs.get("context") or call_args[1].get("context", "")
        assert "calm" in context_str
        assert "Support" in context_str


class TestTTSIntegration:
    """TTS synthesis wires into Audio model correctly."""

    def test_synthesize_returns_audio(self) -> None:
        engine = create_mocked_engine()
        engine._tts.synthesize = AsyncMock(return_value=make_synthesis_result())

        audio = asyncio.run(
            engine.synthesize_speech("Hello", emotion="joyful")
        )
        assert isinstance(audio, Audio)
        assert audio.data == b"RIFF fake audio data"
        assert audio.format == "wav"

    def test_emotion_passed_to_tts(self) -> None:
        engine = create_mocked_engine()
        engine._tts.synthesize = AsyncMock(return_value=make_synthesis_result())

        asyncio.run(
            engine.synthesize_speech("Test", emotion="angry")
        )
        call_kwargs = engine._tts.synthesize.call_args
        assert call_kwargs.kwargs["emotion"] == "angry"


class TestConstitutionalFilterIntegration:
    """Constitutional filter integrates with engine evaluate_intent."""

    def test_engine_with_no_filter_allows(self) -> None:
        engine = create_mocked_engine()
        decision = engine.evaluate_intent("delete_files", [make_span_features()])
        assert decision.allow is True

    def test_engine_with_filter_evaluates(self) -> None:
        engine = create_mocked_engine()
        rules = [
            ConstitutionalRule(
                name="block_delete",
                triggers=["delete"],
                forbidden_prosody=ProsodyCondition(emotion=["sarcastic"]),
            ),
        ]
        engine._filter = ConstitutionalFilter(rules)

        # Sarcastic emotion should trigger forbidden prosody
        decision = engine.evaluate_intent(
            "delete_files",
            [make_span_features()],
            emotion="sarcastic",
        )
        assert decision.allow is False


class TestProfileIntegration:
    """Profile system integrates with engine pipeline."""

    def test_most_specific_mapping_decides_emotion(self) -> None:
        engine = create_mocked_engine()
        alignments, features = make_flat_speech()
        engine._stt.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text="I am fine thank you today.", alignments=alignments, language="en"
            )
        )
        engine._analyzer.analyze = MagicMock(return_value=features)
        engine._analyzer.detect_pauses = MagicMock(return_value=[])

        engine.set_profile(
            ProsodyProfile(
                profile_version="1.0.0",
                user_id="test",
                description=None,
                mappings=(
                    ProsodyMapping(
                        pattern={"pitch_contour": "flat"},
                        interpretation_emotion="calm",
                        confidence_boost=0.6,
                    ),
                    ProsodyMapping(
                        pattern={"pitch_contour": "flat", "rate": "normal"},
                        interpretation_emotion="sincere",
                        confidence_boost=0.7,
                    ),
                ),
            )
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake")
            path = f.name

        try:
            result = asyncio.run(engine.process_voice_input(path))
        finally:
            Path(path).unlink()

        assert (result.emotion, result.confidence) == ("sincere", 0.7)
        assert_valid_iml(result.iml)


class TestCacheIntegration:
    """Cache interacts correctly with the pipeline."""

    def test_cache_hit_skips_pipeline(self) -> None:
        engine = create_mocked_engine()

        engine._stt.transcribe = AsyncMock(
            return_value=make_transcription_result()
        )
        engine._analyzer.analyze = MagicMock(return_value=[])
        engine._analyzer.detect_pauses = MagicMock(return_value=[])
        engine._assembler.assemble = MagicMock(return_value=make_iml_document())
        engine._parser.to_iml_string = MagicMock(return_value="<iml/>")
        engine._validator.validate = MagicMock(return_value=ValidationResult(valid=True))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake")
            path = f.name

        try:
            # First call processes
            r1 = asyncio.run(
                engine.process_voice_input(path)
            )
            assert engine._stt.transcribe.call_count == 1

            # Second call hits cache
            r2 = asyncio.run(
                engine.process_voice_input(path)
            )
            assert engine._stt.transcribe.call_count == 1  # not called again
            assert r1.text == r2.text
        finally:
            Path(path).unlink()
