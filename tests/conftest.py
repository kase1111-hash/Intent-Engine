"""Shared test fixtures and helpers for Intent Engine tests.

Provides the IML validation gate, reusable mock factories, and
sample data for integration and end-to-end tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prosody_protocol import (
    IMLDocument,
    IMLParser,
    IMLValidator,
    ProsodyMapping,
    ProsodyProfile,
    Segment,
    SpanFeatures,
    Utterance,
    ValidationResult,
    WordAlignment,
)

from intent_engine.engine import IntentEngine
from intent_engine.llm.base import InterpretationResult
from intent_engine.models.audio import Audio
from intent_engine.models.response import Response
from intent_engine.models.result import Result
from intent_engine.stt.base import TranscriptionResult
from intent_engine.tts.base import SynthesisResult


# ---------------------------------------------------------------------------
# IML validation gate
# ---------------------------------------------------------------------------

_validator = IMLValidator()


def assert_valid_iml(iml_string: str) -> None:
    """Assert that an IML string passes ``prosody_protocol.IMLValidator``.

    This is the IML validation gate described in Phase 10.  Use this
    in any test that produces IML output.
    """
    result = _validator.validate(iml_string)
    assert result.valid, f"IML validation errors: {result.issues}"


@pytest.fixture()
def iml_validator() -> IMLValidator:
    """Provide a shared IMLValidator instance."""
    return _validator


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------


def make_span_features(
    f0_mean: float | None = 180.0,
    intensity_mean: float | None = 65.0,
    speech_rate: float | None = 4.5,
    quality: str | None = None,
    text: str = "hello",
) -> SpanFeatures:
    """Create a SpanFeatures instance with sensible defaults."""
    return SpanFeatures(
        start_ms=0,
        end_ms=1000,
        text=text,
        f0_mean=f0_mean,
        intensity_mean=intensity_mean,
        speech_rate=speech_rate,
        quality=quality,
    )


def make_iml_document() -> IMLDocument:
    """Create a minimal valid IMLDocument."""
    return IMLDocument(
        utterances=(Utterance(children=(Segment(),)),),
        version="1.0",
    )


def make_transcription_result(
    text: str = "Hello world",
    language: str = "en",
) -> TranscriptionResult:
    """Create a mock TranscriptionResult."""
    return TranscriptionResult(text=text, alignments=[], language=language)


def make_interpretation_result(
    intent: str = "greet",
    response_text: str = "Hello! How can I help you?",
    suggested_emotion: str = "joyful",
) -> InterpretationResult:
    """Create a mock InterpretationResult."""
    return InterpretationResult(
        intent=intent,
        response_text=response_text,
        suggested_emotion=suggested_emotion,
    )


def make_synthesis_result(
    audio_data: bytes = b"RIFF fake audio data",
    format: str = "wav",
    sample_rate: int = 22050,
    duration: float = 1.5,
) -> SynthesisResult:
    """Create a mock SynthesisResult."""
    return SynthesisResult(
        audio_data=audio_data,
        format=format,
        sample_rate=sample_rate,
        duration=duration,
    )


def make_prosody_profile() -> ProsodyProfile:
    """Create a sample prosody profile for testing."""
    return ProsodyProfile(
        profile_version="1.0",
        user_id="test-user",
        description="Test profile for unit tests",
        mappings=[
            ProsodyMapping(
                pattern={"f0_mean": "high"},
                interpretation_emotion="joyful",
                confidence_boost=0.15,
            ),
            ProsodyMapping(
                pattern={"speech_rate": "slow", "quality": "breathy"},
                interpretation_emotion="calm",
                confidence_boost=0.1,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Mocked IntentEngine factory
# ---------------------------------------------------------------------------


def create_mocked_engine(**kwargs) -> IntentEngine:
    """Create an IntentEngine with all providers mocked.

    Returns an engine where STT, LLM, and TTS providers are MagicMock
    instances whose async methods can be configured by tests.
    """
    with patch("intent_engine.engine.create_stt_provider") as stt_f, \
         patch("intent_engine.engine.create_llm_provider") as llm_f, \
         patch("intent_engine.engine.create_tts_provider") as tts_f:
        stt_f.return_value = MagicMock()
        llm_f.return_value = MagicMock()
        tts_f.return_value = MagicMock()
        engine = IntentEngine(**kwargs)
    return engine
