"""IML contract tests: every IML-producing path validates via IMLValidator.

These tests ensure that all Intent Engine code paths that produce IML
output conform to the Prosody Protocol specification.  The shared
``assert_valid_iml`` gate is defined in ``conftest.py``.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from prosody_protocol import (
    IMLDocument,
    IMLParser,
    IMLValidator,
    Segment,
    SpanFeatures,
    Utterance,
    ValidationResult,
)

from intent_engine.engine import IntentEngine
from tests.conftest import (
    assert_valid_iml,
    create_mocked_engine,
    make_iml_document,
    make_span_features,
    make_transcription_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_parser = IMLParser()


def _setup_pipeline_for_iml(engine, iml_string: str = "<iml><utterance>Hello</utterance></iml>"):
    """Configure mocks so process_voice_input returns a specific IML string."""
    engine._stt.transcribe = AsyncMock(
        return_value=make_transcription_result(text="Hello")
    )
    engine._analyzer.analyze = MagicMock(return_value=[make_span_features()])
    engine._analyzer.detect_pauses = MagicMock(return_value=[])
    engine._assembler.assemble = MagicMock(return_value=make_iml_document())
    engine._parser.to_iml_string = MagicMock(return_value=iml_string)
    engine._validator.validate = MagicMock(return_value=ValidationResult(valid=True))
    engine._emotion_classifier.classify = MagicMock(return_value=("neutral", 0.6))


# ---------------------------------------------------------------------------
# IML string validation via the shared gate
# ---------------------------------------------------------------------------


class TestIMLValidationGate:
    """The assert_valid_iml helper itself works correctly."""

    def test_valid_iml_passes(self) -> None:
        assert_valid_iml("<iml><utterance>Hello</utterance></iml>")

    def test_standalone_utterance_passes(self) -> None:
        assert_valid_iml("<utterance>How are you?</utterance>")

    def test_utterance_with_emotion(self) -> None:
        assert_valid_iml(
            '<iml><utterance emotion="joyful" confidence="0.85">Great!</utterance></iml>'
        )

    def test_utterance_with_pause(self) -> None:
        assert_valid_iml(
            '<iml><utterance>Hello<pause duration="300"/>world</utterance></iml>'
        )

    def test_utterance_with_emphasis(self) -> None:
        assert_valid_iml(
            '<iml><utterance><emphasis level="strong">really</emphasis> important</utterance></iml>'
        )

    def test_utterance_with_prosody(self) -> None:
        assert_valid_iml(
            '<iml><utterance><prosody pitch="high">Yes!</prosody></utterance></iml>'
        )

    def test_multi_utterance(self) -> None:
        assert_valid_iml(
            "<iml>"
            "<utterance>First sentence.</utterance>"
            "<utterance>Second sentence.</utterance>"
            "</iml>"
        )

    def test_invalid_iml_fails(self) -> None:
        """Malformed IML should not pass the validation gate."""
        with pytest.raises(AssertionError, match="IML validation errors"):
            assert_valid_iml("<not-iml>broken</not-iml>")


# ---------------------------------------------------------------------------
# Parser round-trip contracts
# ---------------------------------------------------------------------------


class TestIMLParserRoundTrip:
    """IMLParser serialization produces valid IML."""

    def test_document_to_string_validates(self) -> None:
        doc = make_iml_document()
        iml_string = _parser.to_iml_string(doc)
        assert_valid_iml(iml_string)

    def test_parse_then_serialize_validates(self) -> None:
        original = "<iml><utterance>Hello world</utterance></iml>"
        doc = _parser.parse(original)
        serialized = _parser.to_iml_string(doc)
        assert_valid_iml(serialized)

    def test_emotion_utterance_round_trip(self) -> None:
        original = '<iml><utterance emotion="joyful" confidence="0.9">Great!</utterance></iml>'
        doc = _parser.parse(original)
        serialized = _parser.to_iml_string(doc)
        assert_valid_iml(serialized)


# ---------------------------------------------------------------------------
# Engine pipeline IML output contracts
# ---------------------------------------------------------------------------


class TestProcessVoiceInputIMLContract:
    """IML produced by process_voice_input validates."""

    def test_result_iml_is_valid(self) -> None:
        """The IML string in a Result must pass validation."""
        engine = create_mocked_engine()
        iml = "<iml><utterance>Hello</utterance></iml>"
        _setup_pipeline_for_iml(engine, iml_string=iml)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake audio")
            path = f.name

        try:
            result = asyncio.run(
                engine.process_voice_input(path)
            )
            assert_valid_iml(result.iml)
        finally:
            Path(path).unlink()

    def test_multi_utterance_result_validates(self) -> None:
        engine = create_mocked_engine()
        iml = (
            "<iml>"
            "<utterance>First sentence.</utterance>"
            "<utterance>Second sentence.</utterance>"
            "</iml>"
        )
        _setup_pipeline_for_iml(engine, iml_string=iml)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake audio")
            path = f.name

        try:
            result = asyncio.run(
                engine.process_voice_input(path)
            )
            assert_valid_iml(result.iml)
        finally:
            Path(path).unlink()

    def test_emotion_annotated_iml_validates(self) -> None:
        engine = create_mocked_engine()
        iml = '<iml><utterance emotion="frustrated" confidence="0.8">I am upset</utterance></iml>'
        _setup_pipeline_for_iml(engine, iml_string=iml)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake audio")
            path = f.name

        try:
            result = asyncio.run(
                engine.process_voice_input(path)
            )
            assert_valid_iml(result.iml)
        finally:
            Path(path).unlink()


# ---------------------------------------------------------------------------
# Validator catches bad IML (negative contract tests)
# ---------------------------------------------------------------------------


class TestIMLValidatorRejectsInvalid:
    """IMLValidator catches malformed or non-conforming IML."""

    _validator = IMLValidator()

    def test_rejects_missing_root(self) -> None:
        result = self._validator.validate("plain text, no xml")
        assert not result.valid

    def test_rejects_unknown_root_element(self) -> None:
        result = self._validator.validate("<speech>Hello</speech>")
        assert not result.valid

    def test_rejects_negative_pause_duration(self) -> None:
        iml = '<iml><utterance>Hello<pause duration="-100"/>world</utterance></iml>'
        result = self._validator.validate(iml)
        # Should have issues (negative duration violates V10)
        has_issues = not result.valid or len(result.issues) > 0
        assert has_issues

    def test_rejects_emotion_without_confidence(self) -> None:
        """V9: confidence is required when emotion is present."""
        iml = '<iml><utterance emotion="joyful">Great!</utterance></iml>'
        result = self._validator.validate(iml)
        has_issues = not result.valid or len(result.issues) > 0
        assert has_issues


# ---------------------------------------------------------------------------
# Core emotion vocabulary contract
# ---------------------------------------------------------------------------


class TestEmotionVocabularyContract:
    """IML with core emotion labels always validates."""

    CORE_EMOTIONS = [
        "neutral", "sincere", "sarcastic", "frustrated", "joyful",
        "uncertain", "angry", "sad", "fearful", "surprised",
        "disgusted", "calm", "empathetic",
    ]

    @pytest.mark.parametrize("emotion", CORE_EMOTIONS)
    def test_core_emotion_produces_valid_iml(self, emotion: str) -> None:
        iml = f'<iml><utterance emotion="{emotion}" confidence="0.8">Test</utterance></iml>'
        assert_valid_iml(iml)
