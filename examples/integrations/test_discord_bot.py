"""Tests for DiscordBotHelper.

The message payloads hold discord.py objects (an ``Embed``, ``AllowedMentions``)
so that ``channel.send(**payload)`` works as is; these tests build them with the
real library and check them against discord.py's own request builder.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from prosody_protocol import AudioProcessingError

from examples.integrations.discord_bot import DiscordBotHelper
from intent_engine.errors import IntentEngineError, STTError
from intent_engine.models.result import Result

discord = pytest.importorskip("discord")

MakeResult = Callable[..., Result]
MockHttpx = Callable[[Callable[[Any], Any]], list[Any]]


def _joyful(make_result: MakeResult, **kwargs: Any) -> Result:
    kwargs.setdefault("text", "Hello world")
    return make_result(emotion="joyful", confidence=0.85, **kwargs)


def _helper(result: Result | Exception, wav: bytes, **kwargs: Any) -> DiscordBotHelper:
    engine = MagicMock()
    if isinstance(result, Exception):
        engine.process_voice_input = AsyncMock(side_effect=result)
    else:
        engine.process_voice_input = AsyncMock(return_value=result)
    kwargs.setdefault("download_func", AsyncMock(return_value=wav))
    return DiscordBotHelper(engine, **kwargs)


# -- Construction --


class TestDiscordConstruction:
    def test_creates_with_engine(self) -> None:
        helper = DiscordBotHelper(MagicMock())
        assert helper._engine is not None

    def test_custom_format_callback(self) -> None:
        def cb(result: Result, user_id: str) -> str:
            return "custom"

        helper = DiscordBotHelper(MagicMock(), format_callback=cb)
        assert helper._format_callback is cb

    def test_custom_download_func(self) -> None:
        dl = AsyncMock(return_value=b"audio")
        helper = DiscordBotHelper(MagicMock(), download_func=dl)
        assert helper._download_func is dl


# -- Message formatting --


class TestFormatMessage:
    def test_with_user_id(self, make_result: MakeResult) -> None:
        result = _joyful(make_result, text="Hi there")
        text = DiscordBotHelper._format_message(result, user_id="123456")
        assert "<@123456>" in text
        assert "joyful" in text
        assert "Hi there" in text

    def test_with_user_name(self, make_result: MakeResult) -> None:
        text = DiscordBotHelper._format_message(
            _joyful(make_result), user_id=None, user_name="Alice"
        )
        assert "**Alice**" in text

    def test_without_user(self, make_result: MakeResult) -> None:
        text = DiscordBotHelper._format_message(_joyful(make_result))
        assert "A user" in text

    def test_user_id_takes_priority(self, make_result: MakeResult) -> None:
        text = DiscordBotHelper._format_message(
            _joyful(make_result), user_id="123", user_name="Alice"
        )
        assert "<@123>" in text
        assert "Alice" not in text

    def test_abstention_is_not_shown_as_a_detection(self, make_result: MakeResult) -> None:
        # ("neutral", 0.0) means the engine reported no emotion.
        text = DiscordBotHelper._format_message(make_result(text="hello"), user_id="1")
        assert text == "<@1> said: hello"


# -- Discord message building --


class TestBuildDiscordMessage:
    def test_basic_message(self) -> None:
        msg = DiscordBotHelper._build_discord_message("hello")
        assert msg["content"] == "hello"
        assert "embed" not in msg

    def test_with_result_has_embed(self, make_result: MakeResult) -> None:
        msg = DiscordBotHelper._build_discord_message("hello", result=_joyful(make_result))
        assert isinstance(msg["embed"], discord.Embed)
        assert len(msg["embed"].fields) == 3

    def test_embed_fields(self, make_result: MakeResult) -> None:
        result = make_result(emotion="sad", confidence=0.7, suggested_tone="empathetic")
        fields = DiscordBotHelper._build_discord_message("text", result=result)["embed"].fields
        assert fields[0].value == "sad"
        assert fields[1].value == "70%"
        assert fields[2].value == "empathetic"

    def test_emotion_color_joyful(self, make_result: MakeResult) -> None:
        msg = DiscordBotHelper._build_discord_message("text", result=_joyful(make_result))
        assert msg["embed"].colour.value == 0xFFD700

    def test_emotion_color_angry(self, make_result: MakeResult) -> None:
        result = make_result(emotion="angry", confidence=0.9)
        msg = DiscordBotHelper._build_discord_message("text", result=result)
        assert msg["embed"].colour.value == 0xFF0000

    def test_unknown_emotion_default_color(self, make_result: MakeResult) -> None:
        result = make_result(emotion="custom_emotion", confidence=0.9)
        msg = DiscordBotHelper._build_discord_message("text", result=result)
        assert msg["embed"].colour.value == 0x808080  # gray fallback

    def test_embed_fields_inline(self, make_result: MakeResult) -> None:
        msg = DiscordBotHelper._build_discord_message("text", result=_joyful(make_result))
        assert all(field.inline for field in msg["embed"].fields)

    def test_abstention_adds_no_embed(self, make_result: MakeResult) -> None:
        msg = DiscordBotHelper._build_discord_message("hello", result=make_result())
        assert "embed" not in msg

    def test_mentions_never_notify_anyone(self, make_result: MakeResult) -> None:
        # A display name or transcript containing @everyone must not ping the server.
        msg = DiscordBotHelper._build_discord_message(
            "@everyone <@&1234> <@5678>", result=_joyful(make_result)
        )
        allowed = msg["allowed_mentions"]
        assert isinstance(allowed, discord.AllowedMentions)
        assert allowed.to_dict() == {"parse": []}

    def test_long_content_is_truncated_to_discords_limit(self, make_result: MakeResult) -> None:
        msg = DiscordBotHelper._build_discord_message("word " * 700, result=_joyful(make_result))
        assert len(msg["content"]) <= 2000

    @pytest.mark.parametrize("words", [3, 700])
    def test_payload_is_accepted_by_discord_py(self, make_result: MakeResult, words: int) -> None:
        # channel.send(**payload) builds its request with exactly this call.
        from discord.http import handle_message_parameters

        for result in (_joyful(make_result), make_result()):
            payload = DiscordBotHelper._build_discord_message("word " * words, result=result)
            params = handle_message_parameters(**payload)
            assert params.payload is not None
            assert len(params.payload["content"]) <= 2000


# -- process_audio_url --


class TestProcessAudioUrl:
    async def test_returns_message_payload(self, make_result: MakeResult, wav_bytes: bytes) -> None:
        download = AsyncMock(return_value=wav_bytes)
        helper = _helper(_joyful(make_result), wav_bytes, download_func=download)

        msg = await helper.process_audio_url(
            "https://cdn.discordapp.com/audio.wav", user_id="123456"
        )

        assert "content" in msg
        assert isinstance(msg["embed"], discord.Embed)
        download.assert_called_once_with("https://cdn.discordapp.com/audio.wav")

    async def test_custom_format_callback(self, make_result: MakeResult, wav_bytes: bytes) -> None:
        helper = _helper(
            make_result(), wav_bytes, format_callback=lambda r, uid: f"Custom for {uid}"
        )

        msg = await helper.process_audio_url("https://cdn.discordapp.com/f.wav", user_id="U1")

        assert "Custom for U1" in msg["content"]

    async def test_abstaining_result_sends_plain_content(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        helper = _helper(make_result(text="hello"), wav_bytes)

        msg = await helper.process_audio_url("https://cdn.discordapp.com/f.wav", user_id="42")

        assert msg["content"] == "<@42> said: hello"
        assert "embed" not in msg

    async def test_temp_file_is_removed(self, make_result: MakeResult, wav_bytes: bytes) -> None:
        paths: list[str] = []

        async def process(path: str) -> Result:
            paths.append(path)
            return make_result()

        helper = _helper(make_result(), wav_bytes)
        helper._engine.process_voice_input = process

        await helper.process_audio_url("https://cdn.discordapp.com/f.wav")

        assert paths
        assert not Path(paths[0]).exists()

    async def test_user_identity_is_not_logged_at_info(
        self, make_result: MakeResult, wav_bytes: bytes, caplog: pytest.LogCaptureFixture
    ) -> None:
        helper = _helper(make_result(), wav_bytes)

        with caplog.at_level(logging.INFO):
            await helper.process_audio_url(
                "https://cdn.discordapp.com/f.wav", user_id="9876543210", user_name="Alice"
            )

        assert "9876543210" not in caplog.text
        assert "Alice" not in caplog.text


class TestProcessAudioUrlErrors:
    """Failures become a generic chat message; internals stay in the log."""

    @pytest.mark.parametrize(
        "exc",
        [
            IntentEngineError("boom"),
            STTError("STT transcription failed: HTTP 401 from https://api.deepgram.com/v1/listen"),
            AudioProcessingError("Cannot read audio file /tmp/tmpabc123/audio.wav (Not audio)"),
            RuntimeError("secret internal detail"),
        ],
        ids=lambda e: type(e).__name__,
    )
    async def test_engine_failure_returns_generic_message(
        self, exc: Exception, wav_bytes: bytes
    ) -> None:
        helper = _helper(exc, wav_bytes)

        msg = await helper.process_audio_url("https://cdn.discordapp.com/f.wav")

        assert "Failed to process" in msg["content"]
        for leaked in ("/tmp", "deepgram", "secret", "401", "boom"):
            assert leaked not in str(msg)

    async def test_non_audio_download_returns_generic_message(self) -> None:
        engine = MagicMock()
        engine.process_voice_input = AsyncMock()
        helper = DiscordBotHelper(engine, download_func=AsyncMock(return_value=b"<html/>"))

        msg = await helper.process_audio_url("https://cdn.discordapp.com/f.wav")

        assert "Failed to process" in msg["content"]
        engine.process_voice_input.assert_not_called()

    async def test_download_failure_returns_generic_message(self) -> None:
        helper = DiscordBotHelper(
            MagicMock(), download_func=AsyncMock(side_effect=RuntimeError("connection reset"))
        )

        msg = await helper.process_audio_url("https://cdn.discordapp.com/f.wav")

        assert "Failed to process" in msg["content"]
        assert "connection reset" not in str(msg)

    async def test_error_payload_is_accepted_by_discord_py(self, wav_bytes: bytes) -> None:
        from discord.http import handle_message_parameters

        msg = await _helper(RuntimeError("x"), wav_bytes).process_audio_url(
            "https://cdn.discordapp.com/f.wav"
        )

        handle_message_parameters(**msg)


class TestDownloadSafety:
    async def test_foreign_hosts_are_refused_without_a_request(
        self, mock_httpx: MockHttpx
    ) -> None:
        httpx = pytest.importorskip("httpx")
        seen = mock_httpx(lambda request: httpx.Response(200, content=b"x"))
        engine = MagicMock()
        engine.process_voice_input = AsyncMock()
        helper = DiscordBotHelper(engine)

        for url in (
            "http://127.0.0.1:8080/internal",
            "https://evil.example/a.wav",
            "http://cdn.discordapp.com/a.wav",
        ):
            msg = await helper.process_audio_url(url)
            assert "Failed to process" in msg["content"]

        assert seen == []
        engine.process_voice_input.assert_not_called()

    async def test_discord_cdn_is_fetched(
        self, make_result: MakeResult, wav_bytes: bytes, mock_httpx: MockHttpx
    ) -> None:
        httpx = pytest.importorskip("httpx")
        seen = mock_httpx(lambda request: httpx.Response(200, content=wav_bytes))
        helper = _helper(make_result(text="hello"), wav_bytes, download_func=None)

        msg = await helper.process_audio_url(
            "https://cdn.discordapp.com/attachments/1/2/voice-message.ogg?ex=a&is=b&hm=c"
        )

        assert len(seen) == 1
        assert "hello" in msg["content"]

    async def test_oversize_download_is_refused(self, mock_httpx: MockHttpx) -> None:
        httpx = pytest.importorskip("httpx")
        mock_httpx(lambda request: httpx.Response(200, content=b"x" * 500))
        engine = MagicMock()
        engine.process_voice_input = AsyncMock()
        helper = DiscordBotHelper(engine, max_download_bytes=100)

        msg = await helper.process_audio_url("https://cdn.discordapp.com/f.wav")

        assert "Failed to process" in msg["content"]
        engine.process_voice_input.assert_not_called()


# -- process_audio_attachment --


class TestProcessAudioAttachment:
    async def test_extracts_url_from_attachment(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        download = AsyncMock(return_value=wav_bytes)
        helper = _helper(make_result(), wav_bytes, download_func=download)

        attachment = MagicMock()
        attachment.url = "https://cdn.discordapp.com/file.wav"

        msg = await helper.process_audio_attachment(attachment, user_id="123")

        download.assert_called_once_with("https://cdn.discordapp.com/file.wav")
        assert "content" in msg

    async def test_string_attachment_fallback(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        download = AsyncMock(return_value=wav_bytes)
        helper = _helper(make_result(), wav_bytes, download_func=download)

        await helper.process_audio_attachment("https://cdn.discordapp.com/f.wav")

        download.assert_called_once_with("https://cdn.discordapp.com/f.wav")
