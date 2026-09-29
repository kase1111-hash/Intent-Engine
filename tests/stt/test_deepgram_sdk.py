"""DeepgramSTT against the real ``deepgram-sdk`` and a fake ``/v1/listen`` server.

The server listens on 127.0.0.1 only; nothing here reaches Deepgram.  The
tests are skipped when the optional ``deepgram-sdk`` extra is not installed
(or is older than 5.0, which has no ``AsyncDeepgramClient``).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from prosody_protocol import IMLAssembler

from intent_engine.errors import STTError
from intent_engine.stt.deepgram import DeepgramSTT

from .helpers import FakeServer, Heartbeat, json_response, slow, write_wav

deepgram = pytest.importorskip("deepgram")
if not hasattr(deepgram, "AsyncDeepgramClient"):
    pytest.skip("deepgram-sdk 5.x-7.x is required (AsyncDeepgramClient)", allow_module_level=True)

_METADATA = {
    "request_id": "9f0f3c1e-0000-0000-0000-000000000000",
    "sha256": "0" * 64,
    "created": "2026-01-01T00:00:00.000Z",
    "duration": 2.1,
    "channels": 1,
    "models": ["nova-2"],
    "model_info": {},
}


def _word(word: str, start: float, end: float, punctuated: str) -> dict[str, object]:
    return {
        "word": word,
        "start": start,
        "end": end,
        "confidence": 0.99,
        "punctuated_word": punctuated,
    }


_TRANSCRIPT = "I need help. Cancel my order please."
_LISTEN_RESPONSE = {
    "metadata": _METADATA,
    "results": {
        "channels": [
            {
                "alternatives": [
                    {
                        "transcript": _TRANSCRIPT,
                        "confidence": 0.99,
                        "words": [
                            _word("i", 0.0, 0.2, "I"),
                            _word("need", 0.2, 0.4, "need"),
                            _word("help", 0.4, 0.8, "help."),
                            _word("cancel", 1.0, 1.3, "Cancel"),
                            _word("my", 1.3, 1.4, "my"),
                            _word("order", 1.4, 1.7, "order"),
                            _word("please", 1.7, 2.1, "please."),
                        ],
                    }
                ]
            }
        ]
    },
}


@pytest.fixture()
def audio_path(tmp_path: Path) -> str:
    return str(write_wav(tmp_path / "speech.wav"))


@pytest.fixture()
def api(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeServer]:
    """A fake Deepgram API; ``AsyncDeepgramClient`` is pointed at it."""
    server = FakeServer(lambda *_: json_response(_LISTEN_RESPONSE))
    real_client = deepgram.AsyncDeepgramClient

    def environment(url: str) -> object:
        # The constructor's fields differ between 5.x and 7.x; point them all at the server.
        fields = inspect.signature(deepgram.DeepgramClientEnvironment).parameters
        websocket_url = url.replace("http", "ws")
        return deepgram.DeepgramClientEnvironment(
            **{n: websocket_url if n in ("production", "agent") else url for n in fields}
        )

    class LocalClient(real_client):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, environment=environment(server.url), **kwargs)

    monkeypatch.setattr(deepgram, "AsyncDeepgramClient", LocalClient)
    with server:
        yield server


class TestDeepgramSTTAgainstSDK:
    def test_transcribes_through_the_sdk(self, api: FakeServer, audio_path: str) -> None:
        result = asyncio.run(DeepgramSTT(api_key="test-key").transcribe(audio_path))

        assert result.text == _TRANSCRIPT
        assert result.language == "en"
        assert [a.word for a in result.alignments] == [
            "I", "need", "help.", "Cancel", "my", "order", "please.",
        ]
        assert (result.alignments[0].start_ms, result.alignments[-1].end_ms) == (0, 2100)

    def test_sends_the_documented_request(self, api: FakeServer, audio_path: str) -> None:
        asyncio.run(
            DeepgramSTT(api_key="test-key", model="nova-3", language="de").transcribe(audio_path)
        )

        method, path, headers, body = api.requests[0]
        url = urlsplit(path)
        query = parse_qs(url.query)
        assert (method, url.path) == ("POST", "/v1/listen")
        assert query["model"] == ["nova-3"]
        assert query["language"] == ["de"]
        assert query["smart_format"] == ["true"]
        assert query["utterances"] == ["true"]
        assert query["punctuate"] == ["true"]
        assert headers["authorization"] == "Token test-key"
        assert body == Path(audio_path).read_bytes()

    def test_iml_keeps_the_sentence_split(self, api: FakeServer, audio_path: str) -> None:
        result = asyncio.run(DeepgramSTT(api_key="test-key").transcribe(audio_path))
        doc = IMLAssembler().assemble(result.alignments, [], [], language=result.language)
        assert len(doc.utterances) == 2

    def test_event_loop_stays_responsive(self, api: FakeServer, audio_path: str) -> None:
        api.handler = slow(json_response(_LISTEN_RESPONSE), 0.5)

        async def scenario() -> int:
            async with Heartbeat() as heartbeat:
                _, ticks = await heartbeat.ticks_during(
                    DeepgramSTT(api_key="test-key").transcribe(audio_path)
                )
            return ticks

        # A free loop ticks ~25 times in 0.5 s; a blocked one not at all.
        assert asyncio.run(scenario()) >= 10

    def test_http_error_surfaces_as_stt_error_with_the_cause(
        self, api: FakeServer, audio_path: str
    ) -> None:
        api.handler = lambda *_: json_response(
            {"err_code": "INVALID_AUTH", "err_msg": "Invalid credentials."}, status=401
        )
        with pytest.raises(STTError, match="401") as info:
            asyncio.run(DeepgramSTT(api_key="wrong").transcribe(audio_path))
        assert "INVALID_AUTH" in str(info.value)
        assert "deepgram-sdk is required" not in str(info.value)

    def test_accepted_response_without_results_is_an_error(
        self, api: FakeServer, audio_path: str
    ) -> None:
        # The API answers this way when the request was queued (callback mode).
        api.handler = lambda *_: json_response({"request_id": "9f0f3c1e"})
        with pytest.raises(STTError, match="no transcription results"):
            asyncio.run(DeepgramSTT(api_key="test-key").transcribe(audio_path))

    def test_silent_audio_gives_an_empty_result(self, api: FakeServer, audio_path: str) -> None:
        silent = {
            "metadata": _METADATA,
            "results": {
                "channels": [{"alternatives": [{"transcript": "", "confidence": 0.0, "words": []}]}]
            },
        }
        api.handler = lambda *_: json_response(silent)
        result = asyncio.run(DeepgramSTT(api_key="test-key").transcribe(audio_path))
        assert (result.text, result.alignments) == ("", [])
