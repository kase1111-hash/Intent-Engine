"""Prosody profiles: validation, and application through the IML assembler.

Profiles use the Prosody Protocol vocabulary (``pitch``, ``pitch_contour``,
``volume``, ``rate``, ``quality``, ``pause_frequency``,
``emphasis_frequency``).  The engine hands the active profile to
``IMLAssembler``, which matches it per utterance against the speaker's
baseline and marks the utterance with ``x-profile``.  Tests on audio run the
real analyzer and assembler; only STT is faked.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import ModuleType

import pytest
from prosody_protocol import ProfileError, ProsodyMapping, ProsodyProfile

from intent_engine.engine import IntentEngine
from tests.conftest import create_mocked_engine, make_prosody_profile


@pytest.fixture()
def audio() -> ModuleType:
    """The synthetic-audio helpers (skips when numpy/parselmouth are missing)."""
    pytest.importorskip("numpy")
    pytest.importorskip("parselmouth")
    from tests import synth_audio

    return synth_audio


def _calm_on_high_pitch(boost: float = 0.1) -> ProsodyProfile:
    return ProsodyProfile(
        profile_version="1.0.0",
        user_id="user-1",
        description=None,
        mappings=(ProsodyMapping({"pitch": "high"}, "calm", boost),),
    )


def _write_profile(path: Path, **overrides: object) -> Path:
    data: dict[str, object] = {
        "profile_version": "1.0.0",
        "user_id": "user-1",
        "prosody_mappings": [
            {
                "pattern": {"pitch": "high"},
                "interpretation": {"emotion": "calm", "confidence_boost": 0.1},
            }
        ],
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _last_utterance(result):
    return result.iml_document.utterances[-1]


class TestProfileAppliedThroughAssembler:
    def test_set_profile_reaches_the_assembler(self) -> None:
        engine = create_mocked_engine()
        profile = _calm_on_high_pitch()

        engine.set_profile(profile)

        assert engine._assembler.profile is profile

    def test_clear_profile_removes_it_from_the_assembler(self) -> None:
        engine = create_mocked_engine()
        engine.set_profile(_calm_on_high_pitch())

        engine.clear_profile()

        assert engine._assembler.profile is None

    def test_profile_path_in_constructor_reaches_the_assembler(self, tmp_path: Path) -> None:
        path = _write_profile(tmp_path / "profile.json")

        engine = create_mocked_engine(prosody_profile=str(path))

        assert engine._assembler.profile is not None
        assert engine._assembler.profile.user_id == "user-1"

    def test_schema_valid_profile_decides_emotion_and_is_marked_in_iml(
        self, audio: ModuleType, tmp_path: Path
    ) -> None:
        wav = tmp_path / "turn.wav"
        aligns = audio.write_recording(wav, [audio.NEUTRAL] * 4 + [audio.EXCITED])
        engine = audio.engine_for(aligns)

        plain = asyncio.run(engine.process_voice_input(str(wav), use_cache=False))
        assert plain.emotion == "joyful"  # what the classifier says without a profile

        engine.set_profile(_calm_on_high_pitch())
        result = asyncio.run(engine.process_voice_input(str(wav), use_cache=False))

        # the profile (higher pitch means calm for this speaker) takes precedence ...
        assert result.emotion == "calm"
        assert result.confidence == pytest.approx(plain.confidence + 0.1)  # the mapping's boost
        assert result.suggested_tone == "calm"
        # ... and the IML the LLM sees says so
        last = _last_utterance(result)
        assert (last.emotion, dict(last.extra_attributes)) == ("calm", {"x-profile": "pitch=high"})
        assert 'x-profile="pitch=high"' in result.iml

    def test_clear_profile_restores_classifier_emotion(
        self, audio: ModuleType, tmp_path: Path
    ) -> None:
        wav = tmp_path / "turn.wav"
        aligns = audio.write_recording(wav, [audio.NEUTRAL] * 4 + [audio.EXCITED])
        engine = audio.engine_for(aligns)
        engine.set_profile(_calm_on_high_pitch())
        engine.clear_profile()

        result = asyncio.run(engine.process_voice_input(str(wav), use_cache=False))

        assert result.emotion == "joyful"
        assert "x-profile" not in result.iml

    def test_profile_applies_to_a_single_utterance(
        self, audio: ModuleType, tmp_path: Path
    ) -> None:
        # a single sentence has no baseline (so no classifier emotion), but the
        # profile still describes it: flat pitch is this speaker's calm
        wav = tmp_path / "one.wav"
        aligns = audio.write_recording(wav, [audio.NEUTRAL])
        engine = audio.engine_for(aligns)
        engine.set_profile(
            ProsodyProfile(
                "1.0.0", "user-1", None,
                (ProsodyMapping({"pitch_contour": "flat"}, "calm", 0.6),),
            )
        )

        result = asyncio.run(engine.process_voice_input(str(wav), use_cache=False))

        assert (result.emotion, result.confidence) == ("calm", 0.6)


class TestProfileValidation:
    def test_legacy_vocabulary_is_rejected_on_set(self) -> None:
        engine = create_mocked_engine()
        legacy = ProsodyProfile(
            "1.0.0", "u", None, (ProsodyMapping({"f0_mean": "high"}, "joyful", 0.2),)
        )

        with pytest.raises(ProfileError, match="P5"):
            engine.set_profile(legacy)

        assert engine._profile is None

    def test_non_semver_version_is_rejected_on_set(self) -> None:
        engine = create_mocked_engine()
        profile = ProsodyProfile(
            "1.0", "u", None, (ProsodyMapping({"pitch": "high"}, "calm", 0.1),)
        )

        with pytest.raises(ProfileError, match="P1"):
            engine.set_profile(profile)

    def test_failed_set_keeps_the_previous_profile(self) -> None:
        engine = create_mocked_engine()
        good = _calm_on_high_pitch()
        engine.set_profile(good)

        with pytest.raises(ProfileError):
            engine.set_profile(ProsodyProfile("1.0.0", "u", None, ()))

        assert engine._profile is good
        assert engine._assembler.profile is good

    def test_non_profile_is_a_type_error(self) -> None:
        engine = create_mocked_engine()

        with pytest.raises(TypeError, match="ProsodyProfile"):
            engine.set_profile({"user_id": "x"})  # type: ignore[arg-type]

    def test_invalid_profile_file_is_rejected_by_the_constructor(self, tmp_path: Path) -> None:
        path = _write_profile(
            tmp_path / "bad.json",
            prosody_mappings=[
                {"pattern": {"f0_mean": "high"}, "interpretation": {"emotion": "joyful"}}
            ],
        )

        with pytest.raises(ProfileError, match="P5"):
            create_mocked_engine(prosody_profile=str(path))

    def test_invalid_profile_file_is_rejected_by_load_profile(self, tmp_path: Path) -> None:
        engine = create_mocked_engine()
        path = _write_profile(tmp_path / "bad.json", profile_version="1.0")

        with pytest.raises(ProfileError, match="P1"):
            engine.load_profile(str(path))

    def test_load_profile_returns_a_valid_profile(self, tmp_path: Path) -> None:
        engine = create_mocked_engine()
        path = _write_profile(tmp_path / "ok.json")

        profile = engine.load_profile(str(path))

        assert engine.validate_profile(profile).valid

    def test_created_profile_with_defaults_validates(self) -> None:
        engine = create_mocked_engine()

        profile = engine.create_profile(
            "user-1",
            [{"pattern": {"pitch_contour": "flat"}, "interpretation_emotion": "calm"}],
        )

        assert engine.validate_profile(profile).valid
        engine.set_profile(profile)  # and the engine accepts its own profile

    def test_created_profile_with_missing_mapping_key_is_a_profile_error(self) -> None:
        engine = create_mocked_engine()

        with pytest.raises(ProfileError, match="interpretation_emotion"):
            engine.create_profile("u", [{"pattern": {"pitch": "high"}}])

    def test_shared_fixture_profile_is_schema_valid(self) -> None:
        engine = create_mocked_engine()
        assert engine.validate_profile(make_prosody_profile()).valid


def test_no_private_label_vocabulary_is_left() -> None:
    assert not hasattr(IntentEngine, "_derive_feature_labels")
