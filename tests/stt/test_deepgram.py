"""Tests for the Deepgram STT adapter."""

from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from prosody_protocol import IMLAssembler, IMLParser, WordAlignment

from intent_engine.errors import STTError
from intent_engine.stt.base import TranscriptionResult
from intent_engine.stt.deepgram import DeepgramSTT


class TestDeepgramSTTConstruction:
    def test_requires_api_key(self) -> None:
        old_val = os.environ.pop("DEEPGRAM_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="Deepgram API key is required"):
                DeepgramSTT()
        finally:
            if old_val is not None:
                os.environ["DEEPGRAM_API_KEY"] = old_val

    def test_accepts_explicit_api_key(self) -> None:
        stt = DeepgramSTT(api_key="test-key-123")
        assert stt._api_key == "test-key-123"

    def test_reads_env_var(self) -> None:
        os.environ["DEEPGRAM_API_KEY"] = "env-key-456"
        try:
            stt = DeepgramSTT()
            assert stt._api_key == "env-key-456"
        finally:
            del os.environ["DEEPGRAM_API_KEY"]

    def test_default_model_and_language(self) -> None:
        stt = DeepgramSTT(api_key="key")
        assert stt._model == "nova-2"
        assert stt._language == "en"

    def test_custom_model_and_language(self) -> None:
        stt = DeepgramSTT(api_key="key", model="nova-3", language="fr")
        assert stt._model == "nova-3"
        assert stt._language == "fr"

    def test_is_stt_provider(self) -> None:
        from intent_engine.stt.base import STTProvider

        stt = DeepgramSTT(api_key="key")
        assert isinstance(stt, STTProvider)


def _word(
    word: str, start: float, end: float, punctuated_word: str | None = None, **extra: Any
) -> SimpleNamespace:
    return SimpleNamespace(
        word=word,
        start=start,
        end=end,
        confidence=0.9,
        punctuated_word=punctuated_word or word,
        **extra,
    )


_HELP_WORDS = [
    _word("i", 0.0, 0.2, "I"),
    _word("need", 0.2, 0.4),
    _word("help", 0.4, 0.8, "help."),
    _word("cancel", 1.0, 1.3, "Cancel"),
    _word("my", 1.3, 1.4),
    _word("order", 1.4, 1.7),
    _word("please", 1.7, 2.1, "please."),
]


def _response(
    words: list[SimpleNamespace] | None = None,
    transcript: str = "I need help. Cancel my order please.",
    detected_language: str | None = None,
) -> SimpleNamespace:
    """A prerecorded response shaped like the SDK's ``ListenV1Response``."""
    alternative = SimpleNamespace(
        transcript=transcript, confidence=0.9, words=_HELP_WORDS if words is None else words
    )
    channel = SimpleNamespace(alternatives=[alternative], detected_language=detected_language)
    return SimpleNamespace(metadata=SimpleNamespace(), results=SimpleNamespace(channels=[channel]))


class _FakeDeepgram:
    """Installs a stand-in ``deepgram`` module exposing the 5.x-7.x client surface."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, response: Any = None) -> None:
        self.response = _response() if response is None else response
        self.error: Exception | None = None
        self.client_kwargs: dict[str, Any] = {}
        self.call_kwargs: dict[str, Any] = {}
        fake = self

        async def transcribe_file(**kwargs: Any) -> Any:
            fake.call_kwargs = kwargs
            if fake.error is not None:
                raise fake.error
            return fake.response

        class AsyncDeepgramClient:
            def __init__(self, **kwargs: Any) -> None:
                fake.client_kwargs = kwargs
                media = SimpleNamespace(transcribe_file=transcribe_file)
                self.listen = SimpleNamespace(v1=SimpleNamespace(media=media))

        module = types.ModuleType("deepgram")
        module.AsyncDeepgramClient = AsyncDeepgramClient  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "deepgram", module)


@pytest.fixture()
def fake_deepgram(monkeypatch: pytest.MonkeyPatch) -> _FakeDeepgram:
    return _FakeDeepgram(monkeypatch)


@pytest.fixture()
def audio_file(tmp_path: Path) -> str:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF fake audio bytes")
    return str(path)


class TestDeepgramSTTTranscribe:
    def test_import_error_without_sdk(self) -> None:
        import sys
        from unittest.mock import patch

        stt = DeepgramSTT(api_key="key")
        with patch.dict(sys.modules, {"deepgram": None}), pytest.raises(
            ImportError, match="deepgram-sdk is required"
        ):
            asyncio.run(
                stt.transcribe("/some/audio.wav")
            )

    def test_installed_but_unsupported_sdk_is_not_reported_as_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # deepgram-sdk 3.x/4.x: the package imports but has no AsyncDeepgramClient.
        monkeypatch.setitem(sys.modules, "deepgram", types.ModuleType("deepgram"))
        stt = DeepgramSTT(api_key="key")
        with pytest.raises(ImportError) as info:
            asyncio.run(stt.transcribe("/some/audio.wav"))
        message = str(info.value)
        assert "AsyncDeepgramClient" in message
        assert "deepgram-sdk>=5,<8" in message
        assert "is required" not in message
        assert isinstance(info.value.__cause__, ImportError)

    def test_missing_file_raises(self, fake_deepgram: _FakeDeepgram) -> None:
        stt = DeepgramSTT(api_key="key")
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            asyncio.run(stt.transcribe("/nonexistent/audio.wav"))

    def test_uses_async_client_with_current_api(
        self, fake_deepgram: _FakeDeepgram, audio_file: str
    ) -> None:
        stt = DeepgramSTT(api_key="secret", model="nova-3", language="fr")
        asyncio.run(stt.transcribe(audio_file))
        assert fake_deepgram.client_kwargs == {"api_key": "secret"}
        assert fake_deepgram.call_kwargs == {
            "request": b"RIFF fake audio bytes",
            "model": "nova-3",
            "language": "fr",
            "smart_format": True,
            "utterances": True,
            "punctuate": True,
        }

    def test_returns_text_language_and_punctuated_words(
        self, fake_deepgram: _FakeDeepgram, audio_file: str
    ) -> None:
        result = asyncio.run(DeepgramSTT(api_key="key").transcribe(audio_file))
        assert isinstance(result, TranscriptionResult)
        assert result.text == "I need help. Cancel my order please."
        assert result.language == "en"
        assert [a.word for a in result.alignments] == [
            "I", "need", "help.", "Cancel", "my", "order", "please.",
        ]
        assert all(isinstance(a, WordAlignment) for a in result.alignments)
        assert (result.alignments[3].start_ms, result.alignments[3].end_ms) == (1000, 1300)

    def test_detected_language_wins_over_configured_language(
        self, fake_deepgram: _FakeDeepgram, audio_file: str
    ) -> None:
        fake_deepgram.response = _response(detected_language="es")
        result = asyncio.run(DeepgramSTT(api_key="key").transcribe(audio_file))
        assert result.language == "es"

    def test_sentence_punctuation_reaches_iml(
        self, fake_deepgram: _FakeDeepgram, audio_file: str
    ) -> None:
        result = asyncio.run(DeepgramSTT(api_key="key").transcribe(audio_file))
        doc = IMLAssembler().assemble(result.alignments, [], [], language=result.language)
        assert len(doc.utterances) == 2
        iml = IMLParser().to_iml_string(doc)
        assert "I need help." in iml
        assert "Cancel my order please." in iml

    def test_keeps_speaker_labels(self, fake_deepgram: _FakeDeepgram, audio_file: str) -> None:
        fake_deepgram.response = _response(
            words=[
                _word("hi", 0.0, 0.3, "Hi.", speaker=0),
                _word("yes", 0.5, 0.8, "Yes.", speaker=1),
            ],
            transcript="Hi. Yes.",
        )
        result = asyncio.run(DeepgramSTT(api_key="key").transcribe(audio_file))
        assert [a.speaker for a in result.alignments] == ["0", "1"]

    def test_silence_gives_empty_result(
        self, fake_deepgram: _FakeDeepgram, audio_file: str
    ) -> None:
        fake_deepgram.response = _response(words=[], transcript="")
        result = asyncio.run(DeepgramSTT(api_key="key").transcribe(audio_file))
        assert result.text == ""
        assert result.alignments == []

    def test_response_without_results_is_an_error(
        self, fake_deepgram: _FakeDeepgram, audio_file: str
    ) -> None:
        # What the SDK returns when the request was accepted for callback delivery.
        fake_deepgram.response = SimpleNamespace(request_id="abc-123")
        with pytest.raises(STTError, match="no transcription results"):
            asyncio.run(DeepgramSTT(api_key="key").transcribe(audio_file))

    def test_sdk_failure_is_stt_error_with_the_real_cause(
        self, fake_deepgram: _FakeDeepgram, audio_file: str
    ) -> None:
        fake_deepgram.error = RuntimeError("status_code: 401, body: Invalid credentials.")
        with pytest.raises(STTError, match="Invalid credentials") as info:
            asyncio.run(DeepgramSTT(api_key="key").transcribe(audio_file))
        assert info.value.__cause__ is fake_deepgram.error
        assert "deepgram-sdk is required" not in str(info.value)

    def test_unusable_word_timings_are_stt_error(
        self, fake_deepgram: _FakeDeepgram, audio_file: str
    ) -> None:
        fake_deepgram.response = _response(words=[_word("late", 2.0, 1.0)])
        with pytest.raises(STTError, match="late"):
            asyncio.run(DeepgramSTT(api_key="key").transcribe(audio_file))


class TestDeepgramResponseParsing:
    """Test that Deepgram response format is correctly parsed into WordAlignments."""

    def test_word_alignment_creation(self) -> None:
        # Verify we can create the expected output types
        wa = WordAlignment(word="hello", start_ms=0, end_ms=500)
        assert wa.word == "hello"
        assert wa.start_ms == 0
        assert wa.end_ms == 500
