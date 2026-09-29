"""Tests for the safe media downloader shared by the integration examples."""

from __future__ import annotations

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

    async def test_declared_size_over_cap_is_refused(
        self, mock_httpx: MockHttpx
    ) -> None:
        mock_httpx(lambda request: httpx.Response(200, content=b"x" * 100))
        with pytest.raises(MediaDownloadError, match="too large"):
            await download_media("https://files.example.com/a", allowed_hosts=ALLOWED, max_bytes=50)

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
