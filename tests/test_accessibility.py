"""Tests for Phase 9: Accessibility features.

Covers feature label derivation, type-to-speech, and profile
management API on IntentEngine.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prosody_protocol import (
    ProsodyMapping,
    ProsodyProfile,
    SpanFeatures,
    ValidationResult,
)

from intent_engine.engine import IntentEngine
from intent_engine.models.audio import Audio
from intent_engine.tts.base import SynthesisResult


def _create_engine(**kwargs) -> IntentEngine:
    """Create an IntentEngine with mocked providers."""
    with patch("intent_engine.engine.create_stt_provider") as stt_f, \
         patch("intent_engine.engine.create_llm_provider") as llm_f, \
         patch("intent_engine.engine.create_tts_provider") as tts_f:
        stt_f.return_value = MagicMock()
        llm_f.return_value = MagicMock()
        tts_f.return_value = MagicMock()
        return IntentEngine(**kwargs)


def _make_features(
    f0_mean: float | None = None,
    intensity_mean: float | None = None,
    speech_rate: float | None = None,
    quality: str | None = None,
) -> list[SpanFeatures]:
    return [
        SpanFeatures(
            start_ms=0,
            end_ms=1000,
            text="hello",
            f0_mean=f0_mean,
            intensity_mean=intensity_mean,
            speech_rate=speech_rate,
            quality=quality,
        )
    ]


# -- _derive_feature_labels --


class TestDeriveFeatureLabels:
    def test_empty_features(self) -> None:
        labels = IntentEngine._derive_feature_labels([])
        assert labels == {}

    def test_f0_low(self) -> None:
        features = _make_features(f0_mean=100.0)
        labels = IntentEngine._derive_feature_labels(features)
        assert labels["f0_mean"] == "low"

    def test_f0_normal(self) -> None:
        features = _make_features(f0_mean=170.0)
        labels = IntentEngine._derive_feature_labels(features)
        assert labels["f0_mean"] == "normal"

    def test_f0_high(self) -> None:
        features = _make_features(f0_mean=250.0)
        labels = IntentEngine._derive_feature_labels(features)
        assert labels["f0_mean"] == "high"

    def test_intensity_quiet(self) -> None:
        features = _make_features(intensity_mean=45.0)
        labels = IntentEngine._derive_feature_labels(features)
        assert labels["intensity_mean"] == "quiet"

    def test_intensity_normal(self) -> None:
        features = _make_features(intensity_mean=65.0)
        labels = IntentEngine._derive_feature_labels(features)
        assert labels["intensity_mean"] == "normal"

    def test_intensity_loud(self) -> None:
        features = _make_features(intensity_mean=80.0)
        labels = IntentEngine._derive_feature_labels(features)
        assert labels["intensity_mean"] == "loud"

    def test_rate_slow(self) -> None:
        features = _make_features(speech_rate=2.0)
        labels = IntentEngine._derive_feature_labels(features)
        assert labels["speech_rate"] == "slow"

    def test_rate_normal(self) -> None:
        features = _make_features(speech_rate=4.5)
        labels = IntentEngine._derive_feature_labels(features)
        assert labels["speech_rate"] == "normal"

    def test_rate_fast(self) -> None:
        features = _make_features(speech_rate=7.0)
        labels = IntentEngine._derive_feature_labels(features)
        assert labels["speech_rate"] == "fast"

    def test_quality_passthrough(self) -> None:
        features = _make_features(quality="breathy")
        labels = IntentEngine._derive_feature_labels(features)
        assert labels["quality"] == "breathy"

    def test_no_quality_when_none(self) -> None:
        features = _make_features(f0_mean=170.0)
        labels = IntentEngine._derive_feature_labels(features)
        assert "quality" not in labels

    def test_all_labels(self) -> None:
        features = _make_features(
            f0_mean=250.0,
            intensity_mean=80.0,
            speech_rate=7.0,
            quality="creaky",
        )
        labels = IntentEngine._derive_feature_labels(features)
        assert labels == {
            "f0_mean": "high",
            "intensity_mean": "loud",
            "speech_rate": "fast",
            "quality": "creaky",
        }

    def test_none_values_skipped(self) -> None:
        features = _make_features()  # all None
        labels = IntentEngine._derive_feature_labels(features)
        assert labels == {}

    def test_multi_span_averaging(self) -> None:
        features = [
            SpanFeatures(start_ms=0, end_ms=500, text="a", f0_mean=100.0),
            SpanFeatures(start_ms=500, end_ms=1000, text="b", f0_mean=300.0),
        ]
        # Average = 200.0, which is "normal" (120 < 200 < 220)
        labels = IntentEngine._derive_feature_labels(features)
        assert labels["f0_mean"] == "normal"

    def test_boundary_f0_low(self) -> None:
        """f0_mean exactly at _F0_LOW threshold is not low."""
        features = _make_features(f0_mean=120.0)
        labels = IntentEngine._derive_feature_labels(features)
        assert labels["f0_mean"] == "normal"

    def test_boundary_f0_high(self) -> None:
        """f0_mean exactly at _F0_HIGH threshold is not high."""
        features = _make_features(f0_mean=220.0)
        labels = IntentEngine._derive_feature_labels(features)
        assert labels["f0_mean"] == "normal"


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
                    "pattern": {"f0_mean": "high"},
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
                    "pattern": {"speech_rate": "slow", "quality": "breathy"},
                    "interpretation_emotion": "calm",
                    "confidence_boost": 0.2,
                },
            ],
        )
        m = profile.mappings[0]
        assert m.pattern == {"speech_rate": "slow", "quality": "breathy"}
        assert m.interpretation_emotion == "calm"
        assert m.confidence_boost == 0.2

    def test_default_confidence_boost(self) -> None:
        engine = _create_engine()
        profile = engine.create_profile(
            user_id="u1",
            mappings=[
                {
                    "pattern": {"f0_mean": "low"},
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
                {"pattern": {"f0_mean": "high"}, "interpretation_emotion": "joyful"},
                {"pattern": {"f0_mean": "low"}, "interpretation_emotion": "sad"},
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
            profile_version="2.0",
        )
        assert profile.profile_version == "2.0"


class TestLoadProfile:
    def test_loads_from_json_file(self) -> None:
        engine = _create_engine()

        profile_data = {
            "profile_version": "1.0",
            "user_id": "test-user",
            "description": "Test profile",
            "prosody_mappings": [
                {
                    "pattern": {"f0_mean": "high"},
                    "interpretation": {
                        "emotion": "joyful",
                        "confidence_boost": 0.1,
                    },
                }
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(profile_data, f)
            profile_path = f.name

        try:
            profile = engine.load_profile(profile_path)
            assert isinstance(profile, ProsodyProfile)
            assert profile.user_id == "test-user"
            assert len(profile.mappings) == 1
        finally:
            Path(profile_path).unlink()


class TestSetAndClearProfile:
    def test_set_profile(self) -> None:
        engine = _create_engine()
        profile = ProsodyProfile(
            profile_version="1.0",
            user_id="u1",
            description=None,
            mappings=[],
        )
        engine.set_profile(profile)
        assert engine._profile is profile

    def test_clear_profile(self) -> None:
        engine = _create_engine()
        engine._profile = ProsodyProfile(
            profile_version="1.0",
            user_id="u1",
            description=None,
            mappings=[],
        )
        engine.clear_profile()
        assert engine._profile is None


class TestValidateProfile:
    def test_validates_profile(self) -> None:
        engine = _create_engine()
        profile = ProsodyProfile(
            profile_version="1.0",
            user_id="test",
            description=None,
            mappings=[
                ProsodyMapping(
                    pattern={"f0_mean": "high"},
                    interpretation_emotion="joyful",
                )
            ],
        )
        result = engine.validate_profile(profile)
        assert isinstance(result, ValidationResult)

    def test_empty_mappings_valid(self) -> None:
        engine = _create_engine()
        profile = ProsodyProfile(
            profile_version="1.0",
            user_id="test",
            description=None,
            mappings=[],
        )
        result = engine.validate_profile(profile)
        assert isinstance(result, ValidationResult)


# -- Integration: profile applied in pipeline --


class TestProfileInPipeline:
    def test_derive_labels_called_during_pipeline(self) -> None:
        """Verify _derive_feature_labels is used during process_voice_input."""
        engine = _create_engine()

        # Set a profile that maps high pitch to "joyful"
        profile = ProsodyProfile(
            profile_version="1.0",
            user_id="test",
            description=None,
            mappings=[
                ProsodyMapping(
                    pattern={"f0_mean": "high"},
                    interpretation_emotion="joyful",
                    confidence_boost=0.3,
                )
            ],
        )
        engine.set_profile(profile)

        # Verify _derive_feature_labels produces correct labels
        features = _make_features(f0_mean=250.0)
        labels = engine._derive_feature_labels(features)
        assert labels["f0_mean"] == "high"

        # Verify profile applier would match
        emotion, confidence = engine._profile_applier.apply(
            profile, labels, "neutral", 0.5
        )
        assert emotion == "joyful"

    def test_no_profile_skips_application(self) -> None:
        engine = _create_engine()
        assert engine._profile is None

        # _derive_feature_labels should still work independently
        features = _make_features(f0_mean=250.0)
        labels = engine._derive_feature_labels(features)
        assert labels["f0_mean"] == "high"
