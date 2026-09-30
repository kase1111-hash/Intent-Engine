"""AssemblyAISTT against the real ``assemblyai`` SDK and a fake API server.

The server listens on 127.0.0.1 only; nothing here reaches AssemblyAI.  The
tests are skipped when the optional ``assemblyai`` extra is not installed.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from typing import Any

import pytest
from prosody_protocol import IMLAssembler

from intent_engine.errors import STTError
from intent_engine.stt.assemblyai import AssemblyAISTT

from .helpers import FakeServer, Heartbeat, json_response, write_wav

aai = pytest.importorskip("assemblyai")

# ``Word.speaker`` exists from assemblyai 0.21; before that the SDK drops the label.
_HAS_SPEAKER = "speaker" in (getattr(aai.Word, "model_fields", None) or aai.Word.__fields__)


def _word(text: str, start: int, end: int, speaker: str | None = None) -> dict[str, Any]:
    return {"text": text, "start": start, "end": end, "confidence": 0.98, "speaker": speaker}


def _transcript(**fields: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "5551722-f677-13e1-9df2-0e3a9a5cd1b4",
        "status": "completed",
        "audio_url": "https://example.invalid/audio.wav",
        "language_code": "en",
        "text": "I need help. Cancel my order please.",
        "words": [
            _word("I", 0, 200),
            _word("need", 200, 400),
            _word("help.", 400, 800),
            _word("Cancel", 1000, 1300),
            _word("my", 1300, 1400),
            _word("order", 1400, 1700),
            _word("please.", 1700, 2100),
        ],
    }
    body.update(fields)
    return body


def _api(transcript: dict[str, Any], delay: float = 0.0) -> Any:
    """A handler for the upload and transcript endpoints the SDK calls."""
    def handler(method: str, path: str, headers: dict[str, str], body: bytes) -> Any:
        time.sleep(delay)
        if path.startswith("/v2/upload"):
            return json_response({"upload_url": "https://example.invalid/upload/1"})
        return json_response(transcript)

    return handler


@pytest.fixture()
def audio_path(tmp_path: Any) -> str:
    return str(write_wav(tmp_path / "speech.wav"))


@pytest.fixture()
def api(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeServer]:
    """A fake AssemblyAI API; the SDK's global settings are pointed at it."""
    server = FakeServer(_api(_transcript()))
    # setattr first so that the adapter's own writes to these are undone afterwards
    monkeypatch.setattr(aai.settings, "api_key", "unset")
    with server:
        monkeypatch.setattr(aai.settings, "base_url", server.url)
        yield server


class TestAssemblyAISTTAgainstSDK:
    def test_transcribes_through_the_sdk(self, api: FakeServer, audio_path: str) -> None:
        result = asyncio.run(AssemblyAISTT(api_key="test-key").transcribe(audio_path))

        assert result.text == "I need help. Cancel my order please."
        assert result.language == "en"
        assert [a.word for a in result.alignments] == [
            "I", "need", "help.", "Cancel", "my", "order", "please.",
        ]
        assert (result.alignments[0].start_ms, result.alignments[-1].end_ms) == (0, 2100)
        assert any(h.get("authorization") == "test-key" for _, _, h, _ in api.requests)

    def test_iml_keeps_the_sentence_split(self, api: FakeServer, audio_path: str) -> None:
        result = asyncio.run(AssemblyAISTT(api_key="test-key").transcribe(audio_path))
        doc = IMLAssembler().assemble(result.alignments, [], [], language=result.language)
        assert len(doc.utterances) == 2

    @pytest.mark.skipif(not _HAS_SPEAKER, reason="assemblyai < 0.21 has no Word.speaker")
    def test_keeps_speaker_labels(self, api: FakeServer, audio_path: str) -> None:
        api.handler = _api(
            _transcript(
                text="Hi. Yes.",
                words=[_word("Hi.", 0, 300, "A"), _word("Yes.", 500, 800, "B")],
            )
        )
        result = asyncio.run(AssemblyAISTT(api_key="test-key").transcribe(audio_path))
        assert [a.speaker for a in result.alignments] == ["A", "B"]

    def test_event_loop_stays_responsive(self, api: FakeServer, audio_path: str) -> None:
        # Upload, submit and poll each take 0.2 s: 0.6 s in which the loop must stay free.
        api.handler = _api(_transcript(), delay=0.2)

        async def scenario() -> int:
            async with Heartbeat() as heartbeat:
                _, ticks = await heartbeat.ticks_during(
                    AssemblyAISTT(api_key="test-key").transcribe(audio_path)
                )
            return ticks

        # A free loop ticks ~30 times in 0.6 s; a blocked one not at all.
        assert asyncio.run(scenario()) >= 10

    def test_failed_transcript_surfaces_as_stt_error(
        self, api: FakeServer, audio_path: str
    ) -> None:
        api.handler = _api(
            {"id": "t1", "status": "error", "error": "Audio file is corrupt", "audio_url": "x"}
        )
        with pytest.raises(STTError, match="Audio file is corrupt"):
            asyncio.run(AssemblyAISTT(api_key="test-key").transcribe(audio_path))

    def test_unusable_word_timings_are_stt_error(self, api: FakeServer, audio_path: str) -> None:
        api.handler = _api(_transcript(words=[_word("late", 2000, 1000)]))
        with pytest.raises(STTError, match="late"):
            asyncio.run(AssemblyAISTT(api_key="test-key").transcribe(audio_path))

    @pytest.mark.parametrize("empty", [{"text": None, "words": None}, {"text": "", "words": []}])
    def test_silent_audio_gives_an_empty_result(
        self, api: FakeServer, audio_path: str, empty: dict[str, Any]
    ) -> None:
        api.handler = _api(_transcript(**empty))
        result = asyncio.run(AssemblyAISTT(api_key="test-key").transcribe(audio_path))
        assert (result.text, result.alignments) == ("", [])

    def test_missing_file_raises(self, api: FakeServer) -> None:
        with pytest.raises(FileNotFoundError):
            asyncio.run(AssemblyAISTT(api_key="test-key").transcribe("/nonexistent/audio.wav"))
