"""Tests for the safe media downloader shared by the integration examples."""

from __future__ import annotations

import asyncio
import gzip
import logging
import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import pytest

from examples.integrations import _common
from examples.integrations._common import MediaDownloadError, download_media

httpx = pytest.importorskip("httpx")

MockHttpx = Callable[[Callable[[Any], Any]], list[Any]]

ALLOWED = ("files.example.com",)


class TestDownloadMedia:
    async def test_returns_body(self, mock_httpx: MockHttpx) -> None:
        mock_httpx(lambda request: httpx.Response(200, content=b"audio-bytes"))
        data = await download_media("https://files.example.com/a.wav", allowed_hosts=ALLOWED)
        assert data == b"audio-bytes"

    async def test_sends_headers_to_allowed_host(
        self, mock_httpx: MockHttpx
    ) -> None:
        seen = mock_httpx(lambda request: httpx.Response(200, content=b"x"))
        await download_media(
            "https://files.example.com/a.wav",
            allowed_hosts=ALLOWED,
            headers={"Authorization": "Bearer secret"},
        )
        assert seen[0].headers["authorization"] == "Bearer secret"

    @pytest.mark.parametrize(
        "url",
        [
            "http://files.example.com/a.wav",  # not https
            "https://evil.example/a.wav",
            "https://files.example.com.evil.example/a.wav",  # look-alike suffix
            "https://evilfiles.example.com.attacker.test/a.wav",
            "https://notfiles.example.com/a.wav",  # not a subdomain
            "https://files.example.com@evil.example/a.wav",  # userinfo trick
            "https://files.example.com\\@evil.example/a.wav",  # backslash before the real host
            "https://127.0.0.1/a.wav",
            "ftp://files.example.com/a.wav",
            "file:///etc/passwd",
            "not a url",
            "",
        ],
    )
    async def test_refuses_url_without_making_a_request(
        self, url: str, mock_httpx: MockHttpx
    ) -> None:
        seen = mock_httpx(lambda request: httpx.Response(200, content=b"x"))
        with pytest.raises(MediaDownloadError):
            await download_media(url, allowed_hosts=ALLOWED, headers={"Authorization": "secret"})
        assert seen == []

    async def test_allows_subdomains_of_an_allowed_host(
        self, mock_httpx: MockHttpx
    ) -> None:
        mock_httpx(lambda request: httpx.Response(200, content=b"x"))
        assert await download_media("https://a.b.files.example.com/x", allowed_hosts=ALLOWED)

    async def test_body_over_cap_is_refused(self, mock_httpx: MockHttpx) -> None:
        mock_httpx(lambda request: httpx.Response(200, content=b"x" * 100))
        with pytest.raises(MediaDownloadError, match="too large"):
            await download_media("https://files.example.com/a", allowed_hosts=ALLOWED, max_bytes=50)

    async def test_declared_size_over_cap_is_refused_before_the_body_is_read(
        self, mock_httpx: MockHttpx
    ) -> None:
        # The streaming count would refuse this body too; what only the
        # Content-Length check does is refuse it without reading any of it.
        pulled: list[bool] = []

        def handler(request: Any) -> Any:
            async def body() -> Any:
                pulled.append(True)
                yield b"x" * 10

            return httpx.Response(200, headers={"content-length": "100"}, content=body())

        mock_httpx(handler)
        with pytest.raises(MediaDownloadError, match="too large"):
            await download_media("https://files.example.com/a", allowed_hosts=ALLOWED, max_bytes=50)
        assert pulled == []

    async def test_streamed_size_over_cap_is_refused(
        self, mock_httpx: MockHttpx
    ) -> None:
        # No Content-Length: the cap has to be enforced while streaming.
        def handler(request: Any) -> Any:
            async def body() -> Any:
                for _ in range(20):
                    yield b"x" * 10

            return httpx.Response(200, content=body())

        mock_httpx(handler)
        with pytest.raises(MediaDownloadError, match="too large"):
            await download_media("https://files.example.com/a", allowed_hosts=ALLOWED, max_bytes=50)

    async def test_http_error_becomes_download_error(
        self, mock_httpx: MockHttpx
    ) -> None:
        mock_httpx(lambda request: httpx.Response(403, content=b"nope"))
        with pytest.raises(MediaDownloadError, match="403"):
            await download_media("https://files.example.com/a", allowed_hosts=ALLOWED)

    async def test_redirects_are_not_followed(
        self, mock_httpx: MockHttpx
    ) -> None:
        seen = mock_httpx(
            lambda request: httpx.Response(302, headers={"location": "https://evil.example/x"})
        )
        with pytest.raises(MediaDownloadError):
            await download_media(
                "https://files.example.com/a",
                allowed_hosts=ALLOWED,
                headers={"Authorization": "Bearer secret"},
            )
        assert len(seen) == 1

    async def test_transport_error_becomes_download_error(
        self, mock_httpx: MockHttpx
    ) -> None:
        def handler(request: Any) -> Any:
            raise httpx.ConnectError("connection refused to 10.0.0.5")

        mock_httpx(handler)
        with pytest.raises(MediaDownloadError) as info:
            await download_media("https://files.example.com/a", allowed_hosts=ALLOWED)
        assert "10.0.0.5" not in str(info.value)

    def test_default_cap_is_bounded(self) -> None:
        assert 0 < _common.DEFAULT_MAX_DOWNLOAD_BYTES <= 100 * 1024 * 1024


def _stream(data: bytes, chunk: int = 16 * 1024) -> Any:
    """A response body that arrives in pieces, as it does from a socket.

    (``httpx.Response(content=bytes)`` decodes the whole body on construction,
    which is not what a network response does.)
    """

    async def body() -> Any:
        for start in range(0, len(data), chunk):
            yield data[start : start + chunk]

    return body()


class TestContentEncoding:
    """The size cap counts the bytes it is given, so it must be given the wire bytes.

    A ``Content-Encoding`` response is decoded chunk by chunk before it is
    counted, and one small read of a gzip bomb decodes to many MiB.
    Recordings are never sent compressed, so an encoded response is refused.
    """

    async def test_asks_for_an_unencoded_body(self, mock_httpx: MockHttpx) -> None:
        seen = mock_httpx(lambda request: httpx.Response(200, content=b"x"))

        await download_media(
            "https://files.example.com/a",
            allowed_hosts=ALLOWED,
            headers={"Authorization": "Bearer t", "accept-encoding": "gzip, br"},
        )

        assert seen[0].headers["accept-encoding"] == "identity"
        assert seen[0].headers.get_list("accept-encoding") == ["identity"]
        assert seen[0].headers["authorization"] == "Bearer t"  # the rest is kept

    @pytest.mark.parametrize("encoding", ["gzip", "deflate", "gzip, gzip", "identity, gzip"])
    async def test_an_encoded_response_is_refused(
        self, encoding: str, mock_httpx: MockHttpx
    ) -> None:
        # Small and harmless, but a decoder would have run on it.
        body = gzip.compress(b"\x00" * 1000)
        mock_httpx(
            lambda request: httpx.Response(
                200, headers={"content-encoding": encoding}, content=_stream(body)
            )
        )

        with pytest.raises(MediaDownloadError, match="Content-Encoding"):
            await download_media(
                "https://files.example.com/a", allowed_hosts=ALLOWED, max_bytes=10_000
            )

    async def test_a_gzip_bomb_is_refused_without_being_decoded(
        self, mock_httpx: MockHttpx
    ) -> None:
        bomb = gzip.compress(b"\x00" * (32 * 1024 * 1024))  # about 32 KiB on the wire
        mock_httpx(
            lambda request: httpx.Response(
                200, headers={"content-encoding": "gzip"}, content=_stream(bomb)
            )
        )

        tracemalloc.start()
        try:
            with pytest.raises(MediaDownloadError, match="Content-Encoding"):
                await download_media(
                    "https://files.example.com/a", allowed_hosts=ALLOWED, max_bytes=1024 * 1024
                )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert peak < 4 * 1024 * 1024

    @pytest.mark.parametrize("encoding", ["", "identity", "Identity"])
    async def test_an_unencoded_response_is_accepted(
        self, encoding: str, mock_httpx: MockHttpx
    ) -> None:
        headers = {"content-encoding": encoding} if encoding else {}
        mock_httpx(lambda request: httpx.Response(200, headers=headers, content=b"audio"))

        data = await download_media("https://files.example.com/a", allowed_hosts=ALLOWED)

        assert data == b"audio"


class TestTotalDeadline:
    async def test_a_slow_drip_is_cut_off(self, mock_httpx: MockHttpx) -> None:
        # Every read arrives well inside the per-read timeout, so only a
        # deadline on the whole download ends this one.
        def handler(request: Any) -> Any:
            async def body() -> Any:
                for _ in range(100):
                    await asyncio.sleep(0.02)
                    yield b"x"

            return httpx.Response(200, content=body())

        mock_httpx(handler)
        started = time.monotonic()

        with pytest.raises(MediaDownloadError, match="timed out"):
            await download_media(
                "https://files.example.com/a", allowed_hosts=ALLOWED, total_timeout=0.2
            )

        assert time.monotonic() - started < 1.5

    async def test_a_download_inside_the_deadline_completes(self, mock_httpx: MockHttpx) -> None:
        mock_httpx(lambda request: httpx.Response(200, content=b"audio"))

        data = await download_media(
            "https://files.example.com/a", allowed_hosts=ALLOWED, total_timeout=5
        )

        assert data == b"audio"

    def test_the_default_deadline_is_bounded(self) -> None:
        assert 0 < _common.DEFAULT_TOTAL_TIMEOUT <= 600


class TestUrlsStayOutOfLogs:
    """A signed CDN URL is a credential; a failed download must not log one.

    ``logger.exception`` prints the whole exception chain, and the httpx
    error at the bottom of it repeats the full URL in its text.
    """

    QUERY = "?ex=66aa&is=66a9&hm=SIGNATUREVALUE0123456789"
    PATH_ID = "ACACCOUNTIDENTIFIER"

    @staticmethod
    def _calls() -> list[Any]:
        # (name, host, coroutine factory) for each example that downloads a URL.
        from unittest.mock import MagicMock

        from examples.integrations.discord_bot import DiscordBotHelper
        from examples.integrations.slack_bot import SlackBotHelper
        from examples.integrations.twilio_voice import TwilioVoiceHandler

        return [
            (
                "cdn.discordapp.com",
                lambda url: DiscordBotHelper(MagicMock()).process_audio_url(url),
            ),
            (
                "files.slack.com",
                lambda url: SlackBotHelper(MagicMock(), bot_token="xoxb-t").process_audio_file(url),
            ),
            (
                "api.twilio.com",
                lambda url: TwilioVoiceHandler(MagicMock()).handle_voice(url),
            ),
        ]

    @pytest.mark.parametrize("index", [0, 1, 2])
    @pytest.mark.parametrize(
        "status", [404, 302], ids=["client-error", "redirect-with-a-signed-location"]
    )
    async def test_a_failed_download_logs_the_host_and_status_only(
        self,
        index: int,
        status: int,
        mock_httpx: MockHttpx,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        host, call = self._calls()[index]
        headers = {"location": f"https://other.example/x{self.QUERY}"} if status == 302 else {}
        mock_httpx(lambda request: httpx.Response(status, headers=headers))
        url = f"https://{host}/files/{self.PATH_ID}/clip.wav{self.QUERY}"

        # The default level: httpx logs each request URL itself at INFO,
        # which deployments handling signed URLs should keep switched off.
        with caplog.at_level(logging.WARNING):
            await call(url)

        assert f"HTTP {status}" in caplog.text  # the cause is still diagnosable
        assert host in caplog.text
        for secret in ("SIGNATUREVALUE", self.PATH_ID, "hm=", "other.example"):
            assert secret not in caplog.text
