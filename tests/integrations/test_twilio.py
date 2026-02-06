"""Tests for TwilioVoiceHandler."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from intent_engine.errors import IntentEngineError
from intent_engine.integrations.twilio import TwilioVoiceHandler
from intent_engine.models.audio import Audio
from intent_engine.models.result import Result


def _make_result(
    text: str = "Hello",
    emotion: str = "neutral",
    confidence: float = 0.7,
    suggested_tone: str = "neutral",
) -> MagicMock:
    r = MagicMock(spec=Result)
    r.text = text
    r.emotion = emotion
    r.confidence = confidence
    r.suggested_tone = suggested_tone
    return r


def _make_audio(url: str | None = "https://audio.example.com/resp.wav") -> MagicMock:
    a = MagicMock(spec=Audio)
    a.url = url
    return a


# -- Construction --


class TestTwilioConstruction:
    def test_creates_with_engine(self) -> None:
        engine = MagicMock()
        handler = TwilioVoiceHandler(engine)
        assert handler._engine is engine

    def test_custom_response_callback(self) -> None:
        cb = lambda result: "custom"
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


# -- Default response --


class TestDefaultResponse:
    def test_frustrated_escalation(self) -> None:
        result = _make_result(emotion="frustrated")
        text = TwilioVoiceHandler._default_response(result)
        assert "frustrated" in text.lower()
        assert "escalate" in text.lower()

    def test_angry_response(self) -> None:
        result = _make_result(emotion="angry")
        text = TwilioVoiceHandler._default_response(result)
        assert "concern" in text.lower()

    def test_generic_response(self) -> None:
        result = _make_result(text="I need help", emotion="neutral")
        text = TwilioVoiceHandler._default_response(result)
        assert "I need help" in text


# -- handle_voice --


class TestHandleVoice:
    def test_successful_pipeline(self) -> None:
        engine = MagicMock()
        result = _make_result()
        audio = _make_audio()
        engine.process_voice_input = AsyncMock(return_value=result)
        engine.synthesize_speech = AsyncMock(return_value=audio)

        download = AsyncMock(return_value=b"RIFF fake audio")
        handler = TwilioVoiceHandler(engine, download_func=download)

        twiml = asyncio.get_event_loop().run_until_complete(
            handler.handle_voice("https://api.twilio.com/recording.wav")
        )

        assert "<Play>" in twiml
        download.assert_called_once_with("https://api.twilio.com/recording.wav")
        engine.process_voice_input.assert_called_once()

    def test_custom_callback_used(self) -> None:
        engine = MagicMock()
        result = _make_result()
        audio = _make_audio()
        engine.process_voice_input = AsyncMock(return_value=result)
        engine.synthesize_speech = AsyncMock(return_value=audio)

        download = AsyncMock(return_value=b"RIFF fake")
        callback = lambda r: "Custom response text"
        handler = TwilioVoiceHandler(engine, response_callback=callback, download_func=download)

        twiml = asyncio.get_event_loop().run_until_complete(
            handler.handle_voice("https://twilio.com/rec.wav")
        )

        # Synthesize is called with custom text
        engine.synthesize_speech.assert_called_once()
        call_args = engine.synthesize_speech.call_args
        assert call_args[0][0] == "Custom response text"

    def test_fallback_on_no_audio_url(self) -> None:
        engine = MagicMock()
        result = _make_result()
        audio = _make_audio(url=None)
        engine.process_voice_input = AsyncMock(return_value=result)
        engine.synthesize_speech = AsyncMock(return_value=audio)

        download = AsyncMock(return_value=b"RIFF fake")
        handler = TwilioVoiceHandler(engine, download_func=download)

        twiml = asyncio.get_event_loop().run_until_complete(
            handler.handle_voice("https://twilio.com/rec.wav")
        )

        # Falls back to <Say> when no audio URL
        assert "<Say>" in twiml

    def test_error_returns_apology(self) -> None:
        engine = MagicMock()
        engine.process_voice_input = AsyncMock(
            side_effect=IntentEngineError("pipeline broke")
        )

        download = AsyncMock(return_value=b"RIFF fake")
        handler = TwilioVoiceHandler(engine, download_func=download)

        twiml = asyncio.get_event_loop().run_until_complete(
            handler.handle_voice("https://twilio.com/rec.wav")
        )

        assert "<Say>" in twiml
        assert "trouble" in twiml.lower()

    def test_form_data_logged(self) -> None:
        engine = MagicMock()
        result = _make_result()
        audio = _make_audio()
        engine.process_voice_input = AsyncMock(return_value=result)
        engine.synthesize_speech = AsyncMock(return_value=audio)

        download = AsyncMock(return_value=b"RIFF fake")
        handler = TwilioVoiceHandler(engine, download_func=download)

        # Should not raise even with form_data
        asyncio.get_event_loop().run_until_complete(
            handler.handle_voice(
                "https://twilio.com/rec.wav",
                form_data={"CallSid": "CA123", "From": "+15551234567"},
            )
        )
