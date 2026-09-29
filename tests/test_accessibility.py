"""Tests for Phase 9: Accessibility features.

Covers type-to-speech and the profile management API on IntentEngine.
Profiles use the Prosody Protocol vocabulary (see
``schemas/prosody-profile.schema.json``); application through the IML
assembler on real audio is exercised in ``test_engine_profiles.py``.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prosody_protocol import (
    ProfileError,
    ProsodyMapping,
    ProsodyProfile,
    ValidationResult,
)

from intent_engine.engine import IntentEngine
from intent_engine.models.audio import Audio
from intent_engine.stt.base import TranscriptionResult
from intent_engine.tts.base import SynthesisResult
from tests.conftest import assert_valid_iml, make_flat_speech, make_prosody_profile


def _create_engine(**kwargs) -> IntentEngine:
    """Create an IntentEngine with mocked providers."""
    with patch("intent_engine.engine.create_stt_provider") as stt_f, \
         patch("intent_engine.engine.create_llm_provider") as llm_f, \
         patch("intent_engine.engine.create_tts_provider") as tts_f:
        stt_f.return_value = MagicMock()
        llm_f.return_value = MagicMock()
        tts_f.return_value = MagicMock()
        return IntentEngine(**kwargs)


# -- type_to_speech --


class TestTypeToSpeech:
    def test_returns_audio(self) -> None:
        engine = _create_engine()
        engine._tts.synthesize = AsyncMock(
            return_value=SynthesisResult(
                audio_data=b"fake audio", format="wav",
                sample_rate=22050, duration=1.0,
            )
        )

        result = asyncio.run(
            engine.type_to_speech("Hello world", emotion="joyful")
        )
        assert isinstance(result, Audio)
        assert result.data == b"fake audio"

    def test_passes_emotion_to_tts(self) -> None:
        engine = _create_engine()
        engine._tts.synthesize = AsyncMock(
            return_value=SynthesisResult(
                audio_data=b"data", format="wav",
                sample_rate=22050, duration=1.0,
            )
        )

        asyncio.run(
            engine.type_to_speech("Test", emotion="sad")
        )

        call_kwargs = engine._tts.synthesize.call_args
        assert call_kwargs.kwargs["emotion"] == "sad"

    def test_default_emotion_is_neutral(self) -> None:
        engine = _create_engine()
        engine._tts.synthesize = AsyncMock(
            return_value=SynthesisResult(
                audio_data=b"data", format="wav",
                sample_rate=22050, duration=1.0,
            )
        )

        asyncio.run(
            engine.type_to_speech("Test")
        )

        call_kwargs = engine._tts.synthesize.call_args
        assert call_kwargs.kwargs["emotion"] == "neutral"

    def test_sync_wrapper(self) -> None:
        engine = _create_engine()
        engine._tts.synthesize = AsyncMock(
            return_value=SynthesisResult(
                audio_data=b"data", format="wav",
                sample_rate=22050, duration=1.0,
            )
        )

        result = engine.type_to_speech_sync("Hello")
        assert isinstance(result, Audio)


# -- Profile management API --


class TestCreateProfile:
    def test_creates_profile(self) -> None:
        engine = _create_engine()
        profile = engine.create_profile(
            user_id="user-123",
            mappings=[
                {
                    "pattern": {"pitch": "high"},
                    "interpretation_emotion": "joyful",
                    "confidence_boost": 0.1,
                },
            ],
        )
        assert isinstance(profile, ProsodyProfile)
        assert profile.user_id == "user-123"
        assert len(profile.mappings) == 1

    def test_mapping_fields(self) -> None:
        engine = _create_engine()
        profile = engine.create_profile(
            user_id="u1",
            mappings=[
                {
                    "pattern": {"rate": "slow", "quality": "breathy"},
                    "interpretation_emotion": "calm",
                    "confidence_boost": 0.2,
                },
            ],
        )
        m = profile.mappings[0]
        assert m.pattern == {"rate": "slow", "quality": "breathy"}
        assert m.interpretation_emotion == "calm"
        assert m.confidence_boost == 0.2

    def test_default_confidence_boost(self) -> None:
        engine = _create_engine()
        profile = engine.create_profile(
            user_id="u1",
            mappings=[
                {
                    "pattern": {"pitch": "low"},
                    "interpretation_emotion": "neutral",
                },
            ],
        )
        assert profile.mappings[0].confidence_boost == 0.0

    def test_multiple_mappings(self) -> None:
        engine = _create_engine()
        profile = engine.create_profile(
            user_id="u1",
            mappings=[
                {"pattern": {"pitch": "high"}, "interpretation_emotion": "joyful"},
                {"pattern": {"pitch": "low"}, "interpretation_emotion": "sad"},
            ],
        )
        assert len(profile.mappings) == 2

    def test_with_description(self) -> None:
        engine = _create_engine()
        profile = engine.create_profile(
            user_id="u1",
            mappings=[],
            description="Test profile for ASD user",
        )
        assert profile.description == "Test profile for ASD user"

    def test_profile_version(self) -> None:
        engine = _create_engine()
        profile = engine.create_profile(
            user_id="u1",
            mappings=[],
            profile_version="2.0.0",
        )
        assert profile.profile_version == "2.0.0"

    def test_default_profile_version_validates(self) -> None:
        engine = _create_engine()
        profile = engine.create_profile(
            user_id="u1",
            mappings=[{"pattern": {"pitch": "high"}, "interpretation_emotion": "joyful"}],
        )
        assert engine.validate_profile(profile).valid

    def test_mapping_without_required_key_is_a_profile_error(self) -> None:
        engine = _create_engine()
        with pytest.raises(ProfileError, match="interpretation_emotion"):
            engine.create_profile("u1", [{"pattern": {"pitch": "high"}}])


class TestLoadProfile:
    @staticmethod
    def _write(data: dict[str, object]) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            return f.name

    def test_loads_from_json_file(self) -> None:
        engine = _create_engine()

        profile_path = self._write({
            "profile_version": "1.0.0",
            "user_id": "test-user",
            "description": "Test profile",
            "prosody_mappings": [
                {
                    "pattern": {"pitch": "high"},
                    "interpretation": {
                        "emotion": "joyful",
                        "confidence_boost": 0.1,
                    },
                }
            ],
        })

        try:
            profile = engine.load_profile(profile_path)
            assert isinstance(profile, ProsodyProfile)
            assert profile.user_id == "test-user"
            assert len(profile.mappings) == 1
            assert engine.validate_profile(profile).valid
        finally:
            Path(profile_path).unlink()

    def test_rejects_invalid_profile(self) -> None:
        engine = _create_engine()

        profile_path = self._write({
            "profile_version": "1.0.0",
            "user_id": "test-user",
            "prosody_mappings": [
                {
                    "pattern": {"f0_mean": "high"},
                    "interpretation": {"emotion": "joyful"},
                }
            ],
        })

        try:
            with pytest.raises(ProfileError, match="P5"):
                engine.load_profile(profile_path)
        finally:
            Path(profile_path).unlink()


class TestSetAndClearProfile:
    def test_set_profile(self) -> None:
        engine = _create_engine()
        profile = make_prosody_profile()
        engine.set_profile(profile)
        assert engine._profile is profile

    def test_set_profile_rejects_invalid_profile(self) -> None:
        engine = _create_engine()
        profile = ProsodyProfile(
            profile_version="1.0.0",
            user_id="u1",
            description=None,
            mappings=(),
        )
        with pytest.raises(ProfileError, match="P3"):
            engine.set_profile(profile)
        assert engine._profile is None

    def test_clear_profile(self) -> None:
        engine = _create_engine()
        engine.set_profile(make_prosody_profile())
        engine.clear_profile()
        assert engine._profile is None


class TestValidateProfile:
    def test_validates_profile(self) -> None:
        engine = _create_engine()
        result = engine.validate_profile(make_prosody_profile())
        assert isinstance(result, ValidationResult)
        assert result.valid

    def test_legacy_vocabulary_is_invalid(self) -> None:
        engine = _create_engine()
        profile = ProsodyProfile(
            profile_version="1.0.0",
            user_id="test",
            description=None,
            mappings=(
                ProsodyMapping(
                    pattern={"f0_mean": "high"},
                    interpretation_emotion="joyful",
                ),
            ),
        )
        result = engine.validate_profile(profile)
        assert not result.valid
        assert {issue.rule for issue in result.issues} == {"P5"}

    def test_empty_mappings_invalid(self) -> None:
        engine = _create_engine()
        profile = ProsodyProfile(
            profile_version="1.0.0",
            user_id="test",
            description=None,
            mappings=(),
        )
        result = engine.validate_profile(profile)
        assert not result.valid
        assert {issue.rule for issue in result.issues} == {"P3"}


# -- Integration: profile applied in pipeline --


def _process_flat_speech(engine: IntentEngine):
    """Run process_voice_input on monotone speech (real assembler)."""
    alignments, features = make_flat_speech()
    engine._stt.transcribe = AsyncMock(
        return_value=TranscriptionResult(
            text="I am fine thank you today.", alignments=alignments, language="en"
        )
    )
    engine._analyzer.analyze = MagicMock(return_value=features)
    engine._analyzer.detect_pauses = MagicMock(return_value=[])

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"RIFF fake")
        audio_path = f.name
    try:
        return asyncio.run(engine.process_voice_input(audio_path))
    finally:
        Path(audio_path).unlink()


class TestProfileInPipeline:
    def test_profile_decides_emotion_in_iml_and_result(self) -> None:
        """A schema-valid profile matches per utterance via the real assembler."""
        engine = _create_engine()
        engine.set_profile(
            engine.create_profile(
                "asd-user",
                [
                    {
                        "pattern": {"pitch_contour": "flat"},
                        "interpretation_emotion": "calm",
                        "confidence_boost": 0.6,
                    }
                ],
            )
        )

        result = _process_flat_speech(engine)

        assert (result.emotion, result.confidence) == ("calm", 0.6)
        assert 'x-profile="pitch_contour=flat"' in result.iml
        assert_valid_iml(result.iml)

    def test_no_profile_leaves_emotion_unreported(self) -> None:
        engine = _create_engine()
        assert engine._profile is None

        result = _process_flat_speech(engine)

        assert (result.emotion, result.confidence) == ("neutral", 0.0)
        assert "x-profile" not in result.iml
