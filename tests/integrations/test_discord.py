"""Tests for DiscordBotHelper."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from intent_engine.errors import IntentEngineError
from intent_engine.integrations.discord import DiscordBotHelper
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


class TestDiscordConstruction:
    def test_creates_with_engine(self) -> None:
        helper = DiscordBotHelper(MagicMock())
        assert helper._engine is not None

    def test_custom_format_callback(self) -> None:
        cb = lambda r, uid: "custom"
        helper = DiscordBotHelper(MagicMock(), format_callback=cb)
        assert helper._format_callback is cb

    def test_custom_download_func(self) -> None:
        dl = AsyncMock(return_value=b"audio")
        helper = DiscordBotHelper(MagicMock(), download_func=dl)
        assert helper._download_func is dl


# -- Message formatting --


class TestFormatMessage:
    def test_with_user_id(self) -> None:
        result = _make_result(text="Hi there", emotion="joyful")
        text = DiscordBotHelper._format_message(result, user_id="123456")
        assert "<@123456>" in text
        assert "joyful" in text
        assert "Hi there" in text

    def test_with_user_name(self) -> None:
        result = _make_result()
        text = DiscordBotHelper._format_message(
            result, user_id=None, user_name="Alice"
        )
        assert "**Alice**" in text

    def test_without_user(self) -> None:
        result = _make_result()
        text = DiscordBotHelper._format_message(result)
        assert "A user" in text

    def test_user_id_takes_priority(self) -> None:
        result = _make_result()
        text = DiscordBotHelper._format_message(
            result, user_id="123", user_name="Alice"
        )
        assert "<@123>" in text
        assert "Alice" not in text


# -- Discord message building --


class TestBuildDiscordMessage:
    def test_basic_message(self) -> None:
        msg = DiscordBotHelper._build_discord_message("hello")
        assert msg["content"] == "hello"
        assert "embed" not in msg

    def test_with_result_has_embed(self) -> None:
        result = _make_result()
        msg = DiscordBotHelper._build_discord_message("hello", result=result)
        assert "embed" in msg
        assert "fields" in msg["embed"]
        assert len(msg["embed"]["fields"]) == 3

    def test_embed_fields(self) -> None:
        result = _make_result(emotion="sad", confidence=0.7, suggested_tone="empathetic")
        msg = DiscordBotHelper._build_discord_message("text", result=result)
        fields = msg["embed"]["fields"]
        assert fields[0]["value"] == "sad"
        assert fields[1]["value"] == "70%"
        assert fields[2]["value"] == "empathetic"

    def test_emotion_color_joyful(self) -> None:
        result = _make_result(emotion="joyful")
        msg = DiscordBotHelper._build_discord_message("text", result=result)
        assert msg["embed"]["color"] == 0xFFD700

    def test_emotion_color_angry(self) -> None:
        result = _make_result(emotion="angry")
        msg = DiscordBotHelper._build_discord_message("text", result=result)
        assert msg["embed"]["color"] == 0xFF0000

    def test_unknown_emotion_default_color(self) -> None:
        result = _make_result(emotion="custom_emotion")
        msg = DiscordBotHelper._build_discord_message("text", result=result)
        assert msg["embed"]["color"] == 0x808080  # gray fallback

    def test_embed_fields_inline(self) -> None:
        result = _make_result()
        msg = DiscordBotHelper._build_discord_message("text", result=result)
        for field in msg["embed"]["fields"]:
            assert field["inline"] is True


# -- process_audio_url --


class TestProcessAudioUrl:
    def test_returns_message_payload(self) -> None:
        engine = MagicMock()
        result = _make_result()
        engine.process_voice_input = AsyncMock(return_value=result)

        download = AsyncMock(return_value=b"RIFF fake audio")
        helper = DiscordBotHelper(engine, download_func=download)

        msg = asyncio.run(
            helper.process_audio_url(
                "https://cdn.discordapp.com/audio.wav",
                user_id="123456",
            )
        )

        assert "content" in msg
        assert "embed" in msg
        download.assert_called_once_with("https://cdn.discordapp.com/audio.wav")

    def test_custom_format_callback(self) -> None:
        engine = MagicMock()
        engine.process_voice_input = AsyncMock(return_value=_make_result())

        download = AsyncMock(return_value=b"RIFF fake")
        cb = lambda r, uid: f"Custom for {uid}"
        helper = DiscordBotHelper(engine, format_callback=cb, download_func=download)

        msg = asyncio.run(
            helper.process_audio_url("https://discord.com/f.wav", user_id="U1")
        )

        assert "Custom for U1" in msg["content"]

    def test_error_returns_error_message(self) -> None:
        engine = MagicMock()
        engine.process_voice_input = AsyncMock(
            side_effect=IntentEngineError("boom")
        )

        download = AsyncMock(return_value=b"RIFF fake")
        helper = DiscordBotHelper(engine, download_func=download)

        msg = asyncio.run(
            helper.process_audio_url("https://discord.com/f.wav")
        )

        assert "Failed to process" in msg["content"]


# -- process_audio_attachment --


class TestProcessAudioAttachment:
    def test_extracts_url_from_attachment(self) -> None:
        engine = MagicMock()
        engine.process_voice_input = AsyncMock(return_value=_make_result())

        download = AsyncMock(return_value=b"RIFF fake")
        helper = DiscordBotHelper(engine, download_func=download)

        attachment = MagicMock()
        attachment.url = "https://cdn.discordapp.com/file.wav"

        msg = asyncio.run(
            helper.process_audio_attachment(attachment, user_id="123")
        )

        download.assert_called_once_with("https://cdn.discordapp.com/file.wav")
        assert "content" in msg

    def test_string_attachment_fallback(self) -> None:
        engine = MagicMock()
        engine.process_voice_input = AsyncMock(return_value=_make_result())

        download = AsyncMock(return_value=b"RIFF fake")
        helper = DiscordBotHelper(engine, download_func=download)

        msg = asyncio.run(
            helper.process_audio_attachment("https://discord.com/f.wav")
        )

        download.assert_called_once_with("https://discord.com/f.wav")
