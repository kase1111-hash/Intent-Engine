"""Tests for SlackBotHelper."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from intent_engine.errors import IntentEngineError
from intent_engine.integrations.slack import SlackBotHelper
from intent_engine.models.result import Result


def _make_result(
    text: str = "Hello world",
    emotion: str = "joyful",
    confidence: float = 0.85,
    suggested_tone: str = "joyful",
) -> MagicMock:
    r = MagicMock(spec=Result)
    r.text = text
    r.emotion = emotion
    r.confidence = confidence
    r.suggested_tone = suggested_tone
    return r


# -- Construction --


class TestSlackConstruction:
    def test_creates_with_engine(self) -> None:
        helper = SlackBotHelper(MagicMock())
        assert helper._engine is not None

    def test_bot_token_stored(self) -> None:
        helper = SlackBotHelper(MagicMock(), bot_token="xoxb-test")
        assert helper._bot_token == "xoxb-test"

    def test_custom_format_callback(self) -> None:
        cb = lambda result, user_id: "custom"
        helper = SlackBotHelper(MagicMock(), format_callback=cb)
        assert helper._format_callback is cb


# -- Message formatting --


class TestFormatMessage:
    def test_with_user_id(self) -> None:
        result = _make_result(text="Hi there", emotion="joyful", confidence=0.85)
        text = SlackBotHelper._format_message(result, user_id="U123")
        assert "<@U123>" in text
        assert "joyful" in text
        assert "Hi there" in text
        assert "85%" in text

    def test_without_user_id(self) -> None:
        result = _make_result()
        text = SlackBotHelper._format_message(result, user_id=None)
        assert "A user" in text

    def test_confidence_percentage(self) -> None:
        result = _make_result(confidence=0.73)
        text = SlackBotHelper._format_message(result, user_id="U1")
        assert "73%" in text


# -- Slack message building --


class TestBuildSlackMessage:
    def test_basic_message(self) -> None:
        msg = SlackBotHelper._build_slack_message("hello")
        assert msg["text"] == "hello"
        assert "channel" not in msg
        assert "blocks" not in msg

    def test_with_channel(self) -> None:
        msg = SlackBotHelper._build_slack_message("hello", channel_id="C123")
        assert msg["channel"] == "C123"

    def test_with_result_has_blocks(self) -> None:
        result = _make_result()
        msg = SlackBotHelper._build_slack_message("hello", result=result)
        assert "blocks" in msg
        assert len(msg["blocks"]) == 2

    def test_blocks_section_type(self) -> None:
        result = _make_result()
        msg = SlackBotHelper._build_slack_message("hello", result=result)
        assert msg["blocks"][0]["type"] == "section"
        assert msg["blocks"][1]["type"] == "context"

    def test_context_block_has_emotion(self) -> None:
        result = _make_result(emotion="sad", suggested_tone="empathetic")
        msg = SlackBotHelper._build_slack_message("text", result=result)
        context_text = msg["blocks"][1]["elements"][0]["text"]
        assert "sad" in context_text
        assert "empathetic" in context_text


# -- process_audio_file --


class TestProcessAudioFile:
    def test_returns_message_payload(self) -> None:
        engine = MagicMock()
        result = _make_result()
        engine.process_voice_input = AsyncMock(return_value=result)

        download = AsyncMock(return_value=b"RIFF fake audio")
        helper = SlackBotHelper(engine, download_func=download)

        msg = asyncio.get_event_loop().run_until_complete(
            helper.process_audio_file(
                "https://files.slack.com/audio.wav",
                channel_id="C123",
                user_id="U456",
            )
        )

        assert "text" in msg
        assert msg["channel"] == "C123"
        assert "blocks" in msg

    def test_custom_format_callback(self) -> None:
        engine = MagicMock()
        result = _make_result()
        engine.process_voice_input = AsyncMock(return_value=result)

        download = AsyncMock(return_value=b"RIFF fake")
        cb = lambda r, uid: f"Custom for {uid}: {r.text}"
        helper = SlackBotHelper(engine, format_callback=cb, download_func=download)

        msg = asyncio.get_event_loop().run_until_complete(
            helper.process_audio_file("https://slack.com/f.wav", user_id="U1")
        )

        assert "Custom for U1" in msg["text"]

    def test_error_returns_error_message(self) -> None:
        engine = MagicMock()
        engine.process_voice_input = AsyncMock(
            side_effect=IntentEngineError("boom")
        )

        download = AsyncMock(return_value=b"RIFF fake")
        helper = SlackBotHelper(engine, download_func=download)

        msg = asyncio.get_event_loop().run_until_complete(
            helper.process_audio_file("https://slack.com/f.wav")
        )

        assert "Failed to process" in msg["text"]

    def test_download_called_with_url(self) -> None:
        engine = MagicMock()
        engine.process_voice_input = AsyncMock(return_value=_make_result())

        download = AsyncMock(return_value=b"RIFF fake")
        helper = SlackBotHelper(engine, bot_token="xoxb-tok", download_func=download)

        asyncio.get_event_loop().run_until_complete(
            helper.process_audio_file("https://slack.com/f.wav")
        )

        download.assert_called_once_with("https://slack.com/f.wav", "xoxb-tok")


# -- handle_file_shared_event --


class TestHandleFileSharedEvent:
    def test_non_audio_returns_none(self) -> None:
        helper = SlackBotHelper(MagicMock())
        event = {"file": {"mimetype": "image/png"}}

        result = asyncio.get_event_loop().run_until_complete(
            helper.handle_file_shared_event(event)
        )
        assert result is None

    def test_no_download_url_returns_none(self) -> None:
        helper = SlackBotHelper(MagicMock())
        event = {
            "file": {"mimetype": "audio/wav", "url_private_download": ""},
            "channel": "C1",
        }

        result = asyncio.get_event_loop().run_until_complete(
            helper.handle_file_shared_event(event)
        )
        assert result is None

    def test_audio_file_processed(self) -> None:
        engine = MagicMock()
        engine.process_voice_input = AsyncMock(return_value=_make_result())

        download = AsyncMock(return_value=b"RIFF fake")
        helper = SlackBotHelper(engine, download_func=download)

        event = {
            "file": {
                "mimetype": "audio/wav",
                "url_private_download": "https://slack.com/file.wav",
            },
            "channel": "C123",
            "user": "U456",
        }

        result = asyncio.get_event_loop().run_until_complete(
            helper.handle_file_shared_event(event)
        )
        assert result is not None
        assert result["channel"] == "C123"
