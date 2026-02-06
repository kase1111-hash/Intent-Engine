"""End-to-end tests: full pipeline with mocked providers.

Tests the complete user journey: audio in -> Result -> Response -> Audio out.
All external providers are mocked but the internal pipeline logic
(prosody analysis, IML assembly/validation, emotion classification,
constitutional filtering) runs with real Prosody Protocol code.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from prosody_protocol import (
    IMLDocument,
    ProsodyMapping,
    ProsodyProfile,
    Segment,
    SpanFeatures,
    Utterance,
    ValidationResult,
)

from intent_engine.constitutional.rules import ConstitutionalRule, ProsodyCondition, Verification
from intent_engine.constitutional.filter import ConstitutionalFilter
from intent_engine.errors import IntentEngineError, LLMError, STTError, TTSError
from intent_engine.models.audio import Audio
from intent_engine.models.response import Response
from intent_engine.models.result import Result
from tests.conftest import (
    assert_valid_iml,
    create_mocked_engine,
    make_iml_document,
    make_interpretation_result,
    make_span_features,
    make_synthesis_result,
    make_transcription_result,
)


def _setup_full_pipeline(engine, text="Hello", emotion="neutral", confidence=0.6):
    """Configure all mocks for a full pipeline pass."""
    engine._stt.transcribe = AsyncMock(
        return_value=make_transcription_result(text=text)
    )
    engine._analyzer.analyze = MagicMock(
        return_value=[make_span_features()]
    )
    engine._analyzer.detect_pauses = MagicMock(return_value=[])

    iml_doc = make_iml_document()
    engine._assembler.assemble = MagicMock(return_value=iml_doc)
    engine._parser.to_iml_string = MagicMock(
        return_value="<iml><utterance>Hello</utterance></iml>"
    )
    engine._validator.validate = MagicMock(
        return_value=ValidationResult(valid=True)
    )
    engine._emotion_classifier.classify = MagicMock(
        return_value=(emotion, confidence)
    )

    engine._llm.interpret = AsyncMock(
        return_value=make_interpretation_result(
            intent="greet",
            response_text="Hello! How can I help you?",
            suggested_emotion="joyful",
        )
    )

    engine._tts.synthesize = AsyncMock(
        return_value=make_synthesis_result()
    )


class TestFullPipeline:
    """Complete user journey: audio -> Result -> Response -> Audio."""

    def test_audio_to_result(self) -> None:
        engine = create_mocked_engine()
        _setup_full_pipeline(engine, text="I need help", emotion="uncertain", confidence=0.7)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake audio")
            path = f.name

        try:
            result = asyncio.get_event_loop().run_until_complete(
                engine.process_voice_input(path)
            )

            assert isinstance(result, Result)
            assert result.text == "I need help"
            assert result.emotion == "uncertain"
            assert result.confidence == 0.7
            assert result.suggested_tone == "uncertain"
        finally:
            Path(path).unlink()

    def test_result_to_response(self) -> None:
        engine = create_mocked_engine()
        _setup_full_pipeline(engine)

        response = asyncio.get_event_loop().run_until_complete(
            engine.generate_response(
                "<utterance>Hello</utterance>",
                tone="empathetic",
            )
        )

        assert isinstance(response, Response)
        assert response.text == "Hello! How can I help you?"
        assert response.emotion == "joyful"

    def test_response_to_audio(self) -> None:
        engine = create_mocked_engine()
        _setup_full_pipeline(engine)

        audio = asyncio.get_event_loop().run_until_complete(
            engine.synthesize_speech("Hello!", emotion="joyful")
        )

        assert isinstance(audio, Audio)
        assert len(audio.data) > 0
        assert audio.format == "wav"
        assert audio.sample_rate == 22050

    def test_full_round_trip(self) -> None:
        """Complete round trip: audio -> process -> generate -> synthesize."""
        engine = create_mocked_engine()
        _setup_full_pipeline(engine, text="Cancel my subscription", emotion="frustrated", confidence=0.85)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake audio")
            path = f.name

        try:
            # Step 1: Process voice input
            result = asyncio.get_event_loop().run_until_complete(
                engine.process_voice_input(path)
            )
            assert result.emotion == "frustrated"

            # Step 2: Generate response using IML
            response = asyncio.get_event_loop().run_until_complete(
                engine.generate_response(result.iml, tone=result.suggested_tone)
            )
            assert isinstance(response, Response)

            # Step 3: Synthesize response speech
            audio = asyncio.get_event_loop().run_until_complete(
                engine.synthesize_speech(response.text, emotion=response.emotion)
            )
            assert isinstance(audio, Audio)
            assert len(audio.data) > 0
        finally:
            Path(path).unlink()


class TestFullPipelineWithFilter:
    """End-to-end with constitutional filter active."""

    def test_blocked_action(self) -> None:
        engine = create_mocked_engine()
        _setup_full_pipeline(engine, text="Delete everything", emotion="sarcastic", confidence=0.9)

        rules = [
            ConstitutionalRule(
                name="block_sarcastic_delete",
                triggers=["delete"],
                forbidden_prosody=ProsodyCondition(emotion=["sarcastic"]),
            ),
        ]
        engine._filter = ConstitutionalFilter(rules)

        decision = engine.evaluate_intent(
            "delete_everything",
            [make_span_features()],
            emotion="sarcastic",
        )
        assert decision.allow is False
        assert decision.denial_reason is not None

    def test_allowed_action(self) -> None:
        engine = create_mocked_engine()
        _setup_full_pipeline(engine, text="Delete my account", emotion="sincere", confidence=0.85)

        rules = [
            ConstitutionalRule(
                name="verify_sincere_delete",
                triggers=["delete"],
                required_prosody=ProsodyCondition(emotion=["sincere"]),
            ),
        ]
        engine._filter = ConstitutionalFilter(rules)

        decision = engine.evaluate_intent(
            "delete_account",
            [make_span_features()],
            emotion="sincere",
        )
        assert decision.allow is True

    def test_verification_required(self) -> None:
        engine = create_mocked_engine()
        _setup_full_pipeline(engine, text="Delete my account", emotion="neutral", confidence=0.6)

        rules = [
            ConstitutionalRule(
                name="verify_delete",
                triggers=["delete"],
                required_prosody=ProsodyCondition(emotion=["sincere"]),
                verification=Verification(method="explicit_confirmation", retries=2),
            ),
        ]
        engine._filter = ConstitutionalFilter(rules)

        decision = engine.evaluate_intent(
            "delete_account",
            [make_span_features()],
            emotion="neutral",
        )
        assert decision.requires_verification is True
        assert decision.verification_method == "explicit_confirmation"


class TestFullPipelineWithProfile:
    """End-to-end with accessibility profile active."""

    def test_profile_adjusts_emotion(self) -> None:
        engine = create_mocked_engine()
        _setup_full_pipeline(engine, text="I'm fine", emotion="neutral", confidence=0.5)

        profile = ProsodyProfile(
            profile_version="1.0",
            user_id="asd-user-001",
            description="Flat affect -- high pitch means joyful",
            mappings=[
                ProsodyMapping(
                    pattern={"f0_mean": "normal"},
                    interpretation_emotion="calm",
                    confidence_boost=0.2,
                ),
            ],
        )
        engine.set_profile(profile)

        features = [make_span_features(f0_mean=170.0)]
        labels = engine._derive_feature_labels(features)
        emotion, confidence = engine._profile_applier.apply(
            profile, labels, "neutral", 0.5
        )
        assert emotion == "calm"
        assert confidence > 0.5


class TestFullPipelineErrorRecovery:
    """End-to-end error handling."""

    def test_stt_failure_raises_stt_error(self) -> None:
        engine = create_mocked_engine()
        engine._stt.transcribe = AsyncMock(
            side_effect=RuntimeError("Microphone not found")
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake")
            path = f.name

        try:
            with pytest.raises(STTError, match="STT transcription failed"):
                asyncio.get_event_loop().run_until_complete(
                    engine.process_voice_input(path)
                )
        finally:
            Path(path).unlink()

    def test_llm_failure_raises_llm_error(self) -> None:
        engine = create_mocked_engine()
        engine._llm.interpret = AsyncMock(
            side_effect=RuntimeError("API rate limit")
        )

        with pytest.raises(LLMError, match="LLM interpretation failed"):
            asyncio.get_event_loop().run_until_complete(
                engine.generate_response("<iml/>")
            )

    def test_tts_failure_raises_tts_error(self) -> None:
        engine = create_mocked_engine()
        engine._tts.synthesize = AsyncMock(
            side_effect=RuntimeError("Voice model not loaded")
        )

        with pytest.raises(TTSError, match="TTS synthesis failed"):
            asyncio.get_event_loop().run_until_complete(
                engine.synthesize_speech("Hello")
            )

    def test_sync_wrappers_propagate_errors(self) -> None:
        engine = create_mocked_engine()
        engine._tts.synthesize = AsyncMock(
            side_effect=RuntimeError("TTS down")
        )

        with pytest.raises(TTSError):
            engine.synthesize_speech_sync("Hello")
