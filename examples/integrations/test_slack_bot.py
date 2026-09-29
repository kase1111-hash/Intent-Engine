"""Tests for SlackBotHelper."""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from prosody_protocol import AudioProcessingError

from examples.integrations.slack_bot import SlackBotHelper
from intent_engine.errors import IntentEngineError, STTError
from intent_engine.models.result import Result

MakeResult = Callable[..., Result]
MockHttpx = Callable[[Callable[[Any], Any]], list[Any]]


def _joyful(make_result: MakeResult, **kwargs: Any) -> Result:
    kwargs.setdefault("text", "Hello world")
    return make_result(emotion="joyful", confidence=0.85, **kwargs)


def _helper(result: Result | Exception, wav: bytes, **kwargs: Any) -> SlackBotHelper:
    engine = MagicMock()
    if isinstance(result, Exception):
        engine.process_voice_input = AsyncMock(side_effect=result)
    else:
        engine.process_voice_input = AsyncMock(return_value=result)
    kwargs.setdefault("download_func", AsyncMock(return_value=wav))
    return SlackBotHelper(engine, **kwargs)


# -- Construction --


class TestSlackConstruction:
    def test_creates_with_engine(self) -> None:
        helper = SlackBotHelper(MagicMock())
        assert helper._engine is not None

    def test_bot_token_stored(self) -> None:
        helper = SlackBotHelper(MagicMock(), bot_token="xoxb-test")
        assert helper._bot_token == "xoxb-test"

    def test_custom_format_callback(self) -> None:
        def cb(result: Result, user_id: str) -> str:
            return "custom"

        helper = SlackBotHelper(MagicMock(), format_callback=cb)
        assert helper._format_callback is cb


# -- Message formatting --


class TestFormatMessage:
    def test_with_user_id(self, make_result: MakeResult) -> None:
        result = _joyful(make_result, text="Hi there")
        text = SlackBotHelper._format_message(result, user_id="U123")
        assert "<@U123>" in text
        assert "joyful" in text
        assert "Hi there" in text
        assert "85%" in text

    def test_without_user_id(self, make_result: MakeResult) -> None:
        text = SlackBotHelper._format_message(_joyful(make_result), user_id=None)
        assert "A user" in text

    def test_confidence_percentage(self, make_result: MakeResult) -> None:
        result = make_result(emotion="joyful", confidence=0.73)
        text = SlackBotHelper._format_message(result, user_id="U1")
        assert "73%" in text

    def test_abstention_is_not_shown_as_a_detection(self, make_result: MakeResult) -> None:
        # ("neutral", 0.0) means the engine reported no emotion.
        text = SlackBotHelper._format_message(make_result(text="hello"), user_id="U1")
        assert text == "<@U1> said: hello"

    def test_transcript_is_escaped_for_mrkdwn(self, make_result: MakeResult) -> None:
        # An unescaped <!channel> would notify everyone in the channel.
        result = make_result(text="AT&T says <!channel> now")
        text = SlackBotHelper._format_message(result, user_id="U1")
        assert "AT&amp;T says &lt;!channel&gt; now" in text
        assert "<!channel>" not in text


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

    def test_with_result_has_blocks(self, make_result: MakeResult) -> None:
        msg = SlackBotHelper._build_slack_message("hello", result=_joyful(make_result))
        assert "blocks" in msg
        assert len(msg["blocks"]) == 2

    def test_blocks_section_type(self, make_result: MakeResult) -> None:
        msg = SlackBotHelper._build_slack_message("hello", result=_joyful(make_result))
        assert msg["blocks"][0]["type"] == "section"
        assert msg["blocks"][1]["type"] == "context"

    def test_context_block_has_emotion(self, make_result: MakeResult) -> None:
        result = make_result(emotion="sad", confidence=0.8, suggested_tone="empathetic")
        msg = SlackBotHelper._build_slack_message("text", result=result)
        context_text = msg["blocks"][1]["elements"][0]["text"]
        assert "sad" in context_text
        assert "empathetic" in context_text

    def test_abstention_adds_no_emotion_block(self, make_result: MakeResult) -> None:
        msg = SlackBotHelper._build_slack_message("hello", result=make_result())
        assert "blocks" not in msg
        assert "0%" not in str(msg)

    def test_long_text_is_truncated_to_slacks_limits(self, make_result: MakeResult) -> None:
        msg = SlackBotHelper._build_slack_message("word " * 700, result=_joyful(make_result))
        assert len(msg["blocks"][0]["text"]["text"]) <= 3000
        assert len(msg["text"]) <= 3000

    def test_blocks_are_accepted_by_slack_sdk(self, make_result: MakeResult) -> None:
        blocks_module = pytest.importorskip("slack_sdk.models.blocks")
        for words in (3, 700):  # a short clip and a long voice memo
            msg = SlackBotHelper._build_slack_message(
                "word " * words, "C123", _joyful(make_result)
            )
            for block in msg["blocks"]:
                blocks_module.Block.parse(block).validate_json()


# -- process_audio_file --


class TestProcessAudioFile:
    async def test_returns_message_payload(self, make_result: MakeResult, wav_bytes: bytes) -> None:
        helper = _helper(_joyful(make_result), wav_bytes)

        msg = await helper.process_audio_file(
            "https://files.slack.com/audio.wav", channel_id="C123", user_id="U456"
        )

        assert "text" in msg
        assert msg["channel"] == "C123"
        assert "blocks" in msg

    async def test_custom_format_callback(self, make_result: MakeResult, wav_bytes: bytes) -> None:
        helper = _helper(
            make_result(),
            wav_bytes,
            format_callback=lambda r, uid: f"Custom for {uid}: {r.text}",
        )

        msg = await helper.process_audio_file("https://files.slack.com/f.wav", user_id="U1")

        assert "Custom for U1" in msg["text"]

    async def test_download_called_with_url(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        download = AsyncMock(return_value=wav_bytes)
        helper = _helper(make_result(), wav_bytes, bot_token="xoxb-tok", download_func=download)

        await helper.process_audio_file("https://files.slack.com/f.wav")

        download.assert_called_once_with("https://files.slack.com/f.wav", "xoxb-tok")

    async def test_abstaining_result_posts_plain_text(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        helper = _helper(make_result(text="hello"), wav_bytes)

        msg = await helper.process_audio_file("https://files.slack.com/f.wav", "C1", "U1")

        assert msg["text"] == "<@U1> said: hello"
        assert "blocks" not in msg

    async def test_temp_file_is_removed(self, make_result: MakeResult, wav_bytes: bytes) -> None:
        from pathlib import Path

        paths: list[str] = []

        async def process(path: str) -> Result:
            paths.append(path)
            return make_result()

        helper = _helper(make_result(), wav_bytes)
        helper._engine.process_voice_input = process

        await helper.process_audio_file("https://files.slack.com/f.wav")

        assert paths
        assert not Path(paths[0]).exists()

    async def test_user_ids_are_not_logged_at_info(
        self, make_result: MakeResult, wav_bytes: bytes, caplog: pytest.LogCaptureFixture
    ) -> None:
        helper = _helper(make_result(), wav_bytes)

        with caplog.at_level(logging.INFO):
            await helper.process_audio_file("https://files.slack.com/f.wav", "C123", "U456")

        assert "U456" not in caplog.text
        assert "C123" not in caplog.text


class TestProcessAudioFileErrors:
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

        msg = await helper.process_audio_file("https://files.slack.com/f.wav", "C1")

        assert "Failed to process" in msg["text"]
        assert msg["channel"] == "C1"
        for leaked in ("/tmp", "deepgram", "secret", "401", "boom"):
            assert leaked not in str(msg)

    async def test_non_audio_download_returns_generic_message(self) -> None:
        # e.g. Slack answers with an HTML login page when the token lacks access.
        engine = MagicMock()
        engine.process_voice_input = AsyncMock()
        helper = SlackBotHelper(
            engine, download_func=AsyncMock(return_value=b"<html>sign in</html>")
        )

        msg = await helper.process_audio_file("https://files.slack.com/f.wav")

        assert "Failed to process" in msg["text"]
        engine.process_voice_input.assert_not_called()

    async def test_download_failure_returns_generic_message(self) -> None:
        helper = SlackBotHelper(
            MagicMock(), download_func=AsyncMock(side_effect=RuntimeError("connection reset"))
        )

        msg = await helper.process_audio_file("https://files.slack.com/f.wav")

        assert "Failed to process" in msg["text"]
        assert "connection reset" not in str(msg)


class TestDownloadSafety:
    async def test_token_is_not_sent_to_other_hosts(self, mock_httpx: MockHttpx) -> None:
        httpx = pytest.importorskip("httpx")
        seen = mock_httpx(lambda request: httpx.Response(200, content=b"x"))
        engine = MagicMock()
        engine.process_voice_input = AsyncMock()
        helper = SlackBotHelper(engine, bot_token="xoxb-SECRET-TOKEN")

        for url in (
            "http://127.0.0.1:8771/anything",
            "https://evil.example/anything",
            "https://files.slack.com.evil.example/anything",
            "http://files.slack.com/anything",  # cleartext would expose the token
        ):
            msg = await helper.process_audio_file(url)
            assert "Failed to process" in msg["text"]

        assert seen == []
        engine.process_voice_input.assert_not_called()

    async def test_token_is_sent_to_slack(
        self, make_result: MakeResult, wav_bytes: bytes, mock_httpx: MockHttpx
    ) -> None:
        httpx = pytest.importorskip("httpx")
        seen = mock_httpx(lambda request: httpx.Response(200, content=wav_bytes))
        helper = _helper(make_result(), wav_bytes, bot_token="xoxb-tok", download_func=None)

        await helper.process_audio_file("https://files.slack.com/files-pri/T1-F1/clip.wav")

        assert [r.headers["authorization"] for r in seen] == ["Bearer xoxb-tok"]

    async def test_no_token_means_no_authorization_header(
        self, make_result: MakeResult, wav_bytes: bytes, mock_httpx: MockHttpx
    ) -> None:
        httpx = pytest.importorskip("httpx")
        seen = mock_httpx(lambda request: httpx.Response(200, content=wav_bytes))
        helper = _helper(make_result(), wav_bytes, download_func=None)

        await helper.process_audio_file("https://files.slack.com/files-pri/T1-F1/clip.wav")

        assert "authorization" not in seen[0].headers

    async def test_oversize_download_is_refused(self, mock_httpx: MockHttpx) -> None:
        httpx = pytest.importorskip("httpx")
        mock_httpx(lambda request: httpx.Response(200, content=b"x" * 500))
        engine = MagicMock()
        engine.process_voice_input = AsyncMock()
        helper = SlackBotHelper(engine, max_download_bytes=100)

        msg = await helper.process_audio_file("https://files.slack.com/f.wav")

        assert "Failed to process" in msg["text"]
        engine.process_voice_input.assert_not_called()


# -- Request signature verification --


class TestVerifySignature:
    SECRET = "8f742231b10e8888abcd99yyyzzz85a5"
    BODY = '{"type":"event_callback"}'

    def _sign(self, timestamp: str, body: str, secret: str | None = None) -> str:
        base = f"v0:{timestamp}:{body}".encode()
        digest = hmac.new((secret or self.SECRET).encode(), base, hashlib.sha256).hexdigest()
        return f"v0={digest}"

    def _headers(self, timestamp: str, signature: str) -> dict[str, str]:
        return {"X-Slack-Request-Timestamp": timestamp, "X-Slack-Signature": signature}

    def test_accepts_a_valid_signature(self) -> None:
        pytest.importorskip("slack_sdk")
        ts = str(int(time.time()))

        assert SlackBotHelper.verify_signature(
            self.BODY, self._headers(ts, self._sign(ts, self.BODY)), self.SECRET
        )

    def test_rejects_a_wrong_secret(self) -> None:
        pytest.importorskip("slack_sdk")
        ts = str(int(time.time()))
        headers = self._headers(ts, self._sign(ts, self.BODY, secret="another-secret"))

        assert not SlackBotHelper.verify_signature(self.BODY, headers, self.SECRET)

    def test_rejects_a_tampered_body(self) -> None:
        pytest.importorskip("slack_sdk")
        ts = str(int(time.time()))
        headers = self._headers(ts, self._sign(ts, self.BODY))

        assert not SlackBotHelper.verify_signature(self.BODY + " ", headers, self.SECRET)

    def test_rejects_a_stale_timestamp(self) -> None:
        pytest.importorskip("slack_sdk")
        ts = str(int(time.time()) - 3600)

        assert not SlackBotHelper.verify_signature(
            self.BODY, self._headers(ts, self._sign(ts, self.BODY)), self.SECRET
        )

    def test_rejects_a_garbage_timestamp(self) -> None:
        pytest.importorskip("slack_sdk")

        assert not SlackBotHelper.verify_signature(
            self.BODY, self._headers("yesterday", "v0=abc"), self.SECRET
        )

    def test_rejects_missing_headers(self) -> None:
        pytest.importorskip("slack_sdk")

        assert not SlackBotHelper.verify_signature(self.BODY, {}, self.SECRET)


# -- handle_file_shared_event --


class TestHandleFileSharedEvent:
    async def test_non_audio_returns_none(self) -> None:
        helper = SlackBotHelper(MagicMock())
        event = {"file": {"mimetype": "image/png"}}

        assert await helper.handle_file_shared_event(event) is None

    async def test_no_download_url_returns_none(self) -> None:
        helper = SlackBotHelper(MagicMock())
        event = {
            "file": {"mimetype": "audio/wav", "url_private_download": ""},
            "channel": "C1",
        }

        assert await helper.handle_file_shared_event(event) is None

    async def test_audio_file_processed(self, make_result: MakeResult, wav_bytes: bytes) -> None:
        helper = _helper(make_result(), wav_bytes)
        event = {
            "file": {
                "mimetype": "audio/wav",
                "url_private_download": "https://files.slack.com/file.wav",
            },
            "channel": "C123",
            "user": "U456",
        }

        result = await helper.handle_file_shared_event(event)

        assert result is not None
        assert result["channel"] == "C123"
