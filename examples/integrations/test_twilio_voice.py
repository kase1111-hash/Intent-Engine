"""Tests for TwilioVoiceHandler."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from prosody_protocol import AudioProcessingError

from examples.integrations.twilio_voice import TwilioVoiceHandler
from intent_engine.errors import IntentEngineError, STTError
from intent_engine.models.audio import Audio
from intent_engine.models.result import Result

MakeResult = Callable[..., Result]
MockHttpx = Callable[[Callable[[Any], Any]], list[Any]]


def _engine(result: Result, audio: Audio | None = None) -> MagicMock:
    engine = MagicMock()
    engine.process_voice_input = AsyncMock(return_value=result)
    # Audio.url is never set by IntentEngine.synthesize_speech(); neither is it here.
    engine.synthesize_speech = AsyncMock(return_value=audio or Audio(data=b"synthesized"))
    return engine


def _verbs(twiml: str) -> list[ET.Element]:
    """Parse *twiml* (which must be well-formed XML) and return its verbs."""
    root = ET.fromstring(twiml)
    assert root.tag == "Response"
    return list(root)


# -- Construction --


class TestTwilioConstruction:
    def test_creates_with_engine(self) -> None:
        engine = MagicMock()
        handler = TwilioVoiceHandler(engine)
        assert handler._engine is engine

    def test_custom_response_callback(self) -> None:
        def cb(result: Result) -> str:
            return "custom"

        handler = TwilioVoiceHandler(MagicMock(), response_callback=cb)
        assert handler._response_callback is cb

    def test_custom_download_func(self) -> None:
        dl = AsyncMock(return_value=b"audio")
        handler = TwilioVoiceHandler(MagicMock(), download_func=dl)
        assert handler._download_func is dl


# -- TwiML building --


class TestBuildTwiml:
    def test_play_with_audio_url(self) -> None:
        twiml = TwilioVoiceHandler._build_twiml(audio_url="https://example.com/a.wav")
        assert "<Play>https://example.com/a.wav</Play>" in twiml
        assert "<Response>" in twiml
        assert "</Response>" in twiml

    def test_say_without_audio_url(self) -> None:
        twiml = TwilioVoiceHandler._build_twiml(text="Hello there")
        assert "<Say>Hello there</Say>" in twiml
        assert "<Play>" not in twiml

    def test_prefers_play_over_say(self) -> None:
        twiml = TwilioVoiceHandler._build_twiml(
            audio_url="https://example.com/a.wav", text="fallback"
        )
        assert "<Play>" in twiml
        assert "<Say>" not in twiml

    def test_xml_declaration(self) -> None:
        twiml = TwilioVoiceHandler._build_twiml(text="test")
        assert twiml.startswith('<?xml version="1.0" encoding="UTF-8"?>')

    def test_ampersand_in_text_is_escaped(self) -> None:
        twiml = TwilioVoiceHandler._build_twiml(text="I have a problem with AT&T billing today")
        (verb,) = _verbs(twiml)  # raises ParseError if not well-formed
        assert verb.tag == "Say"
        assert verb.text == "I have a problem with AT&T billing today"

    def test_markup_in_text_cannot_inject_verbs(self) -> None:
        hostile = "hello </Say><Dial>+19005551234</Dial><Say> bye"
        (verb,) = _verbs(TwilioVoiceHandler._build_twiml(text=hostile))
        assert verb.tag == "Say"
        assert verb.text == hostile
        assert list(verb) == []

    def test_url_with_query_string_is_escaped(self) -> None:
        url = "https://b.s3.amazonaws.com/a.mp3?X-Amz-Expires=300&X-Amz-Signature=abc"
        twiml = TwilioVoiceHandler._build_twiml(audio_url=url)
        assert "&amp;X-Amz-Signature" in twiml
        (verb,) = _verbs(twiml)
        assert verb.tag == "Play"
        assert verb.text == url

    def test_characters_invalid_in_xml_are_dropped(self) -> None:
        twiml = TwilioVoiceHandler._build_twiml(text="a\x00b\x0bc\x1fd")
        (verb,) = _verbs(twiml)
        assert verb.text == "abcd"


# -- Default response --


class TestDefaultResponse:
    def test_frustrated_escalation(self, make_result: MakeResult) -> None:
        result = make_result(emotion="frustrated", confidence=0.8)
        text = TwilioVoiceHandler._default_response(result)
        assert "frustrated" in text.lower()
        assert "escalate" in text.lower()

    def test_angry_response(self, make_result: MakeResult) -> None:
        result = make_result(emotion="angry", confidence=0.8)
        text = TwilioVoiceHandler._default_response(result)
        assert "concern" in text.lower()

    def test_generic_response(self, make_result: MakeResult) -> None:
        result = make_result(text="I need help", emotion="neutral", confidence=0.7)
        text = TwilioVoiceHandler._default_response(result)
        assert "I need help" in text

    def test_abstention_never_triggers_an_emotion_branch(self, make_result: MakeResult) -> None:
        # ("neutral", 0.0): the engine reported no emotion.
        text = TwilioVoiceHandler._default_response(make_result(text="I need help"))
        assert text == "I heard: I need help. Let me help you with that."

    def test_low_confidence_anger_is_not_acted_on(self, make_result: MakeResult) -> None:
        result = make_result(text="I need help", emotion="angry", confidence=0.2)
        assert "escalate" not in TwilioVoiceHandler._default_response(result).lower()
        assert "concern" not in TwilioVoiceHandler._default_response(result).lower()


# -- handle_voice --


class TestHandleVoice:
    async def test_without_publisher_says_the_reply_and_skips_tts(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        # IntentEngine never sets Audio.url, so without somewhere to publish the
        # synthesized bytes there is nothing <Play> could point at.  Do not pay
        # for a synthesis that would be thrown away.
        engine = _engine(make_result(text="Hello there"))
        download = AsyncMock(return_value=wav_bytes)
        handler = TwilioVoiceHandler(engine, download_func=download)

        twiml = await handler.handle_voice("https://api.twilio.com/recording")

        (verb,) = _verbs(twiml)
        assert verb.tag == "Say"
        assert "Hello there" in (verb.text or "")
        engine.synthesize_speech.assert_not_called()
        download.assert_called_once_with("https://api.twilio.com/recording")
        engine.process_voice_input.assert_called_once()

    async def test_with_publisher_plays_the_published_url(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        result = make_result(emotion="angry", confidence=0.9)
        audio = Audio(data=b"synthesized", format="mp3")
        engine = _engine(result, audio)
        publisher = AsyncMock(return_value="https://cdn.example.com/reply.mp3?a=1&b=2")
        handler = TwilioVoiceHandler(
            engine,
            download_func=AsyncMock(return_value=wav_bytes),
            audio_publisher=publisher,
        )

        twiml = await handler.handle_voice("https://api.twilio.com/recording")

        (verb,) = _verbs(twiml)
        assert verb.tag == "Play"
        assert verb.text == "https://cdn.example.com/reply.mp3?a=1&b=2"
        publisher.assert_awaited_once_with(audio)
        args, kwargs = engine.synthesize_speech.call_args
        assert "concern" in args[0].lower()
        assert kwargs["emotion"] == result.suggested_tone

    async def test_publisher_failure_falls_back_to_say(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        engine = _engine(make_result(text="Hello there"))
        publisher = AsyncMock(side_effect=RuntimeError("bucket unavailable"))
        handler = TwilioVoiceHandler(
            engine, download_func=AsyncMock(return_value=wav_bytes), audio_publisher=publisher
        )

        (verb,) = _verbs(await handler.handle_voice("https://api.twilio.com/recording"))

        assert verb.tag == "Say"
        assert "Hello there" in (verb.text or "")

    async def test_tts_failure_falls_back_to_say(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        engine = _engine(make_result(text="Hello there"))
        engine.synthesize_speech = AsyncMock(side_effect=IntentEngineError("tts down"))
        handler = TwilioVoiceHandler(
            engine,
            download_func=AsyncMock(return_value=wav_bytes),
            audio_publisher=AsyncMock(return_value="https://cdn.example.com/x.mp3"),
        )

        (verb,) = _verbs(await handler.handle_voice("https://api.twilio.com/recording"))

        assert verb.tag == "Say"

    async def test_custom_callback_used(self, make_result: MakeResult, wav_bytes: bytes) -> None:
        engine = _engine(make_result())
        handler = TwilioVoiceHandler(
            engine,
            response_callback=lambda r: "Custom response text",
            download_func=AsyncMock(return_value=wav_bytes),
            audio_publisher=AsyncMock(return_value="https://cdn.example.com/x.mp3"),
        )

        await handler.handle_voice("https://api.twilio.com/rec")

        assert engine.synthesize_speech.call_args[0][0] == "Custom response text"

    async def test_transcript_with_markup_stays_well_formed(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        engine = _engine(make_result(text="AT&T </Say><Dial>+19005551234</Dial><Say>"))
        handler = TwilioVoiceHandler(engine, download_func=AsyncMock(return_value=wav_bytes))

        verbs = _verbs(await handler.handle_voice("https://api.twilio.com/rec"))

        assert [v.tag for v in verbs] == ["Say"]

    async def test_form_data_accepted(self, make_result: MakeResult, wav_bytes: bytes) -> None:
        engine = _engine(make_result())
        handler = TwilioVoiceHandler(engine, download_func=AsyncMock(return_value=wav_bytes))

        await handler.handle_voice(
            "https://api.twilio.com/rec",
            form_data={"CallSid": "CA123", "From": "+15551234567"},
        )

    async def test_temp_file_is_removed(self, make_result: MakeResult, wav_bytes: bytes) -> None:
        from pathlib import Path

        engine = _engine(make_result())
        paths: list[str] = []

        async def process(path: str) -> Result:
            paths.append(path)
            assert Path(path).is_file()
            return make_result()

        engine.process_voice_input = process
        handler = TwilioVoiceHandler(engine, download_func=AsyncMock(return_value=wav_bytes))

        await handler.handle_voice("https://api.twilio.com/rec")

        assert paths
        assert not Path(paths[0]).exists()


class TestHandleVoiceErrors:
    """Failures become a spoken apology; nothing internal reaches the caller."""

    @pytest.mark.parametrize(
        "exc",
        [
            IntentEngineError("pipeline broke"),
            STTError("STT transcription failed: HTTP 401 from https://api.deepgram.com/v1/listen"),
            AudioProcessingError("Cannot read audio file /tmp/tmpabc123/audio.wav (Not audio)"),
            RuntimeError("secret internal detail"),
        ],
        ids=lambda e: type(e).__name__,
    )
    async def test_engine_failure_returns_apology(self, exc: Exception, wav_bytes: bytes) -> None:
        engine = MagicMock()
        engine.process_voice_input = AsyncMock(side_effect=exc)
        handler = TwilioVoiceHandler(engine, download_func=AsyncMock(return_value=wav_bytes))

        twiml = await handler.handle_voice("https://api.twilio.com/rec")

        (verb,) = _verbs(twiml)
        assert verb.tag == "Say"
        assert "trouble" in twiml.lower()
        for leaked in ("/tmp", "deepgram", "secret", "401"):
            assert leaked not in twiml

    async def test_undecodable_download_returns_apology_without_calling_engine(self) -> None:
        engine = MagicMock()
        engine.process_voice_input = AsyncMock()
        handler = TwilioVoiceHandler(
            engine, download_func=AsyncMock(return_value=b"<html>not audio</html>")
        )

        twiml = await handler.handle_voice("https://api.twilio.com/rec")

        assert "trouble" in twiml.lower()
        engine.process_voice_input.assert_not_called()

    async def test_download_failure_returns_apology(self) -> None:
        engine = MagicMock()
        handler = TwilioVoiceHandler(
            engine, download_func=AsyncMock(side_effect=RuntimeError("connection reset"))
        )

        twiml = await handler.handle_voice("https://api.twilio.com/rec")

        assert "trouble" in twiml.lower()
        assert "connection reset" not in twiml

    async def test_default_downloader_refuses_foreign_hosts(self, mock_httpx: MockHttpx) -> None:
        httpx = pytest.importorskip("httpx")
        seen = mock_httpx(lambda request: httpx.Response(200, content=b"x"))
        engine = MagicMock()
        engine.process_voice_input = AsyncMock()
        handler = TwilioVoiceHandler(engine)

        twiml = await handler.handle_voice("http://127.0.0.1:8080/internal")

        assert "trouble" in twiml.lower()
        assert seen == []
        engine.process_voice_input.assert_not_called()

    async def test_default_downloader_fetches_twilio_recordings(
        self, make_result: MakeResult, wav_bytes: bytes, mock_httpx: MockHttpx
    ) -> None:
        httpx = pytest.importorskip("httpx")
        seen = mock_httpx(lambda request: httpx.Response(200, content=wav_bytes))
        engine = _engine(make_result(text="Hello there"))
        handler = TwilioVoiceHandler(engine)

        twiml = await handler.handle_voice(
            "https://api.twilio.com/2010-04-01/Accounts/AC1/Recordings/RE1"
        )

        assert [str(r.url) for r in seen] == [
            "https://api.twilio.com/2010-04-01/Accounts/AC1/Recordings/RE1"
        ]
        assert "Hello there" in twiml

    async def test_default_downloader_enforces_size_cap(self, mock_httpx: MockHttpx) -> None:
        httpx = pytest.importorskip("httpx")
        mock_httpx(lambda request: httpx.Response(200, content=b"x" * 200))
        engine = MagicMock()
        engine.process_voice_input = AsyncMock()
        handler = TwilioVoiceHandler(engine, max_download_bytes=100)

        twiml = await handler.handle_voice("https://api.twilio.com/rec")

        assert "trouble" in twiml.lower()
        engine.process_voice_input.assert_not_called()


# -- Signature validation --


class TestValidateTwilioSignature:
    def test_accepts_a_valid_signature(self) -> None:
        pytest.importorskip("twilio")
        from twilio.request_validator import RequestValidator

        url = "https://example.com/voice"
        params = {"CallSid": "CA123", "RecordingUrl": "https://api.twilio.com/rec"}
        signature = RequestValidator("token").compute_signature(url, params)

        assert TwilioVoiceHandler.validate_twilio_signature(url, params, signature, "token")

    def test_rejects_a_forged_signature(self) -> None:
        pytest.importorskip("twilio")

        assert not TwilioVoiceHandler.validate_twilio_signature(
            "https://example.com/voice", {"CallSid": "CA123"}, "forged", "token"
        )
