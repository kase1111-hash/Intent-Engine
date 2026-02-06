"""Tests for the TTS base interface, emotion mapping, and SynthesisResult."""

from __future__ import annotations

import asyncio

import pytest

from intent_engine.tts.base import (
    EMOTION_VOICE_MAP,
    EmotionVoiceParams,
    SynthesisResult,
    TTSProvider,
    get_voice_params,
)


class TestEmotionVoiceParams:
    def test_construction(self) -> None:
        params = EmotionVoiceParams(
            pitch_shift="+10%", rate=1.15, volume_db=2.0,
            style_notes="Bright, upbeat",
        )
        assert params.pitch_shift == "+10%"
        assert params.rate == 1.15
        assert params.volume_db == 2.0
        assert params.style_notes == "Bright, upbeat"

    def test_frozen(self) -> None:
        params = EmotionVoiceParams(
            pitch_shift="0%", rate=1.0, volume_db=0.0,
            style_notes="Default",
        )
        with pytest.raises(AttributeError):
            params.rate = 2.0  # type: ignore[misc]


class TestEmotionVoiceMap:
    CORE_EMOTIONS = [
        "neutral", "sincere", "sarcastic", "frustrated", "joyful",
        "uncertain", "angry", "sad", "fearful", "surprised",
        "disgusted", "calm", "empathetic",
    ]

    def test_all_core_emotions_present(self) -> None:
        for emotion in self.CORE_EMOTIONS:
            assert emotion in EMOTION_VOICE_MAP, f"Missing emotion: {emotion}"

    def test_has_thirteen_entries(self) -> None:
        assert len(EMOTION_VOICE_MAP) == 13

    def test_all_values_are_emotion_voice_params(self) -> None:
        for emotion, params in EMOTION_VOICE_MAP.items():
            assert isinstance(params, EmotionVoiceParams), (
                f"EMOTION_VOICE_MAP[{emotion!r}] is not EmotionVoiceParams"
            )

    def test_neutral_is_baseline(self) -> None:
        neutral = EMOTION_VOICE_MAP["neutral"]
        assert neutral.rate == 1.0
        assert neutral.volume_db == 0.0
        assert neutral.pitch_shift == "0%"

    def test_angry_is_loud_and_fast(self) -> None:
        angry = EMOTION_VOICE_MAP["angry"]
        assert angry.rate > 1.0
        assert angry.volume_db > 0.0

    def test_sad_is_slow_and_quiet(self) -> None:
        sad = EMOTION_VOICE_MAP["sad"]
        assert sad.rate < 1.0
        assert sad.volume_db < 0.0


class TestGetVoiceParams:
    def test_known_emotion(self) -> None:
        params = get_voice_params("joyful")
        assert params == EMOTION_VOICE_MAP["joyful"]

    def test_unknown_emotion_falls_back_to_neutral(self) -> None:
        params = get_voice_params("completely_unknown_emotion")
        assert params == EMOTION_VOICE_MAP["neutral"]

    def test_returns_emotion_voice_params(self) -> None:
        params = get_voice_params("empathetic")
        assert isinstance(params, EmotionVoiceParams)


class TestSynthesisResult:
    def test_construction(self) -> None:
        result = SynthesisResult(audio_data=b"fake audio")
        assert result.audio_data == b"fake audio"
        assert result.format == "wav"
        assert result.sample_rate == 22050
        assert result.duration is None

    def test_with_all_fields(self) -> None:
        result = SynthesisResult(
            audio_data=b"audio bytes",
            format="mp3",
            sample_rate=44100,
            duration=2.5,
        )
        assert result.format == "mp3"
        assert result.sample_rate == 44100
        assert result.duration == 2.5

    def test_frozen(self) -> None:
        result = SynthesisResult(audio_data=b"data")
        with pytest.raises(AttributeError):
            result.format = "ogg"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = SynthesisResult(audio_data=b"abc", format="wav")
        b = SynthesisResult(audio_data=b"abc", format="wav")
        assert a == b


class TestTTSProviderInterface:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            TTSProvider()  # type: ignore[abstract]

    def test_subclass_must_implement_synthesize(self) -> None:
        class IncompleteTTS(TTSProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteTTS()  # type: ignore[abstract]

    def test_concrete_subclass(self) -> None:
        class ConcreteTTS(TTSProvider):
            async def synthesize(
                self, text: str, emotion: str = "neutral", **kwargs: object
            ) -> SynthesisResult:
                return SynthesisResult(audio_data=b"audio")

        tts = ConcreteTTS()
        assert isinstance(tts, TTSProvider)

        result = asyncio.run(
            tts.synthesize("Hello", emotion="calm")
        )
        assert isinstance(result, SynthesisResult)
        assert result.audio_data == b"audio"
