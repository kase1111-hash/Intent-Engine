"""ElevenLabsTTS against the real ``elevenlabs`` SDK and a fake local server.

The SDK is pointed at an HTTP server bound to 127.0.0.1, so these tests check
what actually goes over the wire and how the adapter behaves while a request
is in flight, without an API key or network access.  They are skipped when
the optional ``elevenlabs`` extra is not installed.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import typing
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from intent_engine.tts.elevenlabs import ElevenLabsTTS
from tests.tts.helpers import SDK_OUTPUT_FORMATS, Heartbeat

elevenlabs = pytest.importorskip("elevenlabs")

AUDIO = b"ID3" + bytes(range(256)) * 4


class FakeElevenLabs:
    """Threaded HTTP server standing in for api.elevenlabs.io."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length)
                url = urlparse(self.path)
                outer.requests.append(
                    {
                        "path": url.path,
                        "query": parse_qs(url.query),
                        "headers": dict(self.headers),
                        "body": json.loads(body),
                    }
                )
                time.sleep(outer.delay)
                self.send_response(200)
                self.send_header("content-type", "audio/mpeg")
                self.send_header("content-length", str(len(AUDIO)))
                self.end_headers()
                self.wfile.write(AUDIO)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture()
def fake_api(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[float], FakeElevenLabs]]:
    """Point every ``elevenlabs.ElevenLabs`` client at a local fake server."""
    servers: list[FakeElevenLabs] = []
    # never route the loopback requests through a proxy configured in the environment
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    # other tests replace and pop sys.modules["elevenlabs"]; make sure the adapter's
    # `from elevenlabs import ElevenLabs` sees the module patched below
    monkeypatch.setitem(sys.modules, "elevenlabs", elevenlabs)

    def start(delay: float = 0.0) -> FakeElevenLabs:
        server = FakeElevenLabs(delay)
        servers.append(server)
        original = elevenlabs.ElevenLabs

        class Redirected(original):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                kwargs["base_url"] = server.base_url
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(elevenlabs, "ElevenLabs", Redirected)
        return server

    yield start
    for server in servers:
        server.close()


async def test_request_shape_and_audio(fake_api: Callable[[float], FakeElevenLabs]) -> None:
    server = fake_api(0.0)
    tts = ElevenLabsTTS(api_key="test-key", voice_id="voice123", model_id="eleven_multilingual_v2")
    result = await tts.synthesize("Hello there", emotion="sad")

    assert result.audio_data == AUDIO
    assert (result.format, result.sample_rate) == ("mp3", 44100)
    (request,) = server.requests
    assert request["path"] == "/v1/text-to-speech/voice123"
    assert request["query"]["output_format"] == ["mp3_44100_128"]
    assert request["headers"]["xi-api-key"] == "test-key"
    assert request["body"]["text"] == "Hello there"
    assert request["body"]["model_id"] == "eleven_multilingual_v2"
    assert request["body"]["voice_settings"]["stability"] == 0.55  # the "sad" preset


@pytest.mark.parametrize(("output_format", "expected"), sorted(SDK_OUTPUT_FORMATS.items()))
async def test_format_and_sample_rate_follow_output_format(
    output_format: str,
    expected: tuple[str, int],
    fake_api: Callable[[float], FakeElevenLabs],
) -> None:
    server = fake_api(0.0)
    result = await ElevenLabsTTS(api_key="k", output_format=output_format).synthesize("Hi")
    assert (result.format, result.sample_rate) == expected
    assert server.requests[0]["query"]["output_format"] == [output_format]


def test_every_format_the_installed_sdk_accepts_is_understood() -> None:
    try:
        from elevenlabs.text_to_speech.types import TextToSpeechConvertRequestOutputFormat
    except ImportError:
        pytest.skip("this elevenlabs version does not expose the output format type")
    literal = typing.get_args(TextToSpeechConvertRequestOutputFormat)[0]
    formats = typing.get_args(literal)
    assert formats, "no output formats found in the SDK"
    for output_format in formats:
        codec, _, rest = output_format.partition("_")
        rate = int(rest.split("_")[0])
        tts = ElevenLabsTTS(api_key="k", output_format=output_format)
        assert (tts._audio_format, tts._sample_rate) == (codec, rate), output_format


async def test_event_loop_stays_responsive_during_the_request(
    fake_api: Callable[[float], FakeElevenLabs],
) -> None:
    fake_api(0.5)
    async with Heartbeat() as heartbeat:
        result = await ElevenLabsTTS(api_key="k").synthesize("Hello there")
    assert result.audio_data == AUDIO
    assert heartbeat.ticks >= 10
