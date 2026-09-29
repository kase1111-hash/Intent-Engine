"""Tests for the TTS base interface, emotion mapping, and SynthesisResult."""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from intent_engine.tts.base import (
    EMOTION_VOICE_MAP,
    EmotionVoiceParams,
    SynthesisResult,
    TTSProvider,
    get_voice_params,
    normalize_emotion,
    strip_ssml,
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


class TestEmotionNormalisation:
    """``get_voice_params`` must never crash on what an LLM hands back."""

    @pytest.mark.parametrize("raw", ["Sad", "SAD", " sad ", "sad\n", "\tSad  "])
    def test_case_and_whitespace_are_normalised(self, raw: str) -> None:
        assert get_voice_params(raw) is EMOTION_VOICE_MAP["sad"]
        assert normalize_emotion(raw) == "sad"

    @pytest.mark.parametrize("emotion", sorted(EMOTION_VOICE_MAP))
    def test_every_core_label_is_kept(self, emotion: str) -> None:
        assert normalize_emotion(emotion) == emotion
        assert get_voice_params(emotion.upper()) is EMOTION_VOICE_MAP[emotion]

    @pytest.mark.parametrize(
        "raw",
        [None, ["calm"], {"emotion": "calm"}, 5, 1.5, b"sad", ("sad",), object(), True],
    )
    def test_non_string_values_fall_back_to_neutral(self, raw: object) -> None:
        assert get_voice_params(raw) is EMOTION_VOICE_MAP["neutral"]
        assert normalize_emotion(raw) == "neutral"

    @pytest.mark.parametrize("raw", ["", "   ", "excited", "warm", "stressed", "sad."])
    def test_unknown_labels_fall_back_to_neutral(self, raw: str) -> None:
        assert get_voice_params(raw) is EMOTION_VOICE_MAP["neutral"]

    def test_unknown_label_is_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="intent_engine.tts.base"):
            normalize_emotion("excited")
        assert "excited" in caplog.text
        assert "neutral" in caplog.text

    def test_non_string_value_is_reported_by_type(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="intent_engine.tts.base"):
            normalize_emotion(["calm"])
        assert "list" in caplog.text

    @pytest.mark.parametrize("raw", ["neutral", "Sad", None, ""])
    def test_no_warning_for_known_or_absent_emotion(
        self, raw: str | None, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="intent_engine.tts.base"):
            normalize_emotion(raw)
        assert caplog.text == ""

    def test_overlong_unknown_label_is_truncated_in_the_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="intent_engine.tts.base"):
            normalize_emotion("x" * 5000)
        assert len(caplog.text) < 500


class TestStripSSML:
    WRAPPED = (
        '<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xml:lang="en-US"><s><prosody pitch="+5%" volume="+3dB">I am so happy!'
        "</prosody></s></speak>"
    )

    def test_wrapped_document_becomes_plain_text(self) -> None:
        assert strip_ssml(self.WRAPPED) == "I am so happy!"

    def test_sentences_and_breaks_keep_word_boundaries(self) -> None:
        ssml = (
            '<speak xml:lang="en-US"><s>Hello <break time="300ms"/>there</s>'
            '<s>Second <emphasis level="strong">one</emphasis></s></speak>'
        )
        assert strip_ssml(ssml) == "Hello there Second one"

    def test_inline_tags_do_not_split_words(self) -> None:
        ssml = "<speak><s>un<emphasis>believ</emphasis>able</s></speak>"
        assert strip_ssml(ssml) == "unbelievable"

    def test_entities_are_decoded(self) -> None:
        assert strip_ssml("<speak><s>Tom &amp; Jerry &lt;3</s></speak>") == "Tom & Jerry <3"

    def test_surrounding_whitespace_and_xml_declaration(self) -> None:
        ssml = '  <?xml version="1.0"?>\n<speak><s>Hi</s></speak>\n'
        assert strip_ssml(ssml) == "Hi"

    def test_empty_document_gives_empty_string(self) -> None:
        assert strip_ssml('<speak xml:lang="en-US"><s></s></speak>') == ""

    @pytest.mark.parametrize(
        "body",
        ["<" * 100_000, "<a" * 100_000, "<>" * 50_000, " <" * 50_000],
        ids=["lt", "lt-letter", "lt-gt", "space-lt"],
    )
    def test_adversarial_input_is_handled_in_linear_time(self, body: str) -> None:
        # Unmatched "<" runs took seconds (quadratic) with a naive tag pattern.
        start = time.perf_counter()
        strip_ssml("<speak>" + body + "</speak>")
        assert time.perf_counter() - start < 1.0

    @pytest.mark.parametrize(
        "text",
        [
            "Hello world",
            "",
            "if a < b and c > d",
            "<b>bold</b> is not SSML",
            "Say <speak> to start",
            "<speak>unterminated",
            "prefix <speak><s>Hi</s></speak>",
        ],
    )
    def test_anything_else_is_returned_unchanged(self, text: str) -> None:
        assert strip_ssml(text) == text


class TestSSMLSupportFlag:
    def test_base_provider_does_not_claim_ssml(self) -> None:
        assert TTSProvider.supports_ssml is False

    def test_concrete_provider_inherits_default(self) -> None:
        class ConcreteTTS(TTSProvider):
            async def synthesize(
                self, text: str, emotion: str = "neutral", **kwargs: object
            ) -> SynthesisResult:
                return SynthesisResult(audio_data=b"audio")

        assert ConcreteTTS().supports_ssml is False
