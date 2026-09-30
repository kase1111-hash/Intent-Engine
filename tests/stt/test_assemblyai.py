"""Tests for the AssemblyAI STT adapter."""

from __future__ import annotations

import asyncio
import enum
import os
import sys
import threading
import time
import types
from types import SimpleNamespace
from typing import Any

import pytest
from prosody_protocol import WordAlignment

from intent_engine.errors import STTError
from intent_engine.stt.assemblyai import AssemblyAISTT

from .helpers import Heartbeat


class TestAssemblyAISTTConstruction:
    def test_requires_api_key(self) -> None:
        old_val = os.environ.pop("ASSEMBLYAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="AssemblyAI API key is required"):
                AssemblyAISTT()
        finally:
            if old_val is not None:
                os.environ["ASSEMBLYAI_API_KEY"] = old_val

    def test_accepts_explicit_api_key(self) -> None:
        stt = AssemblyAISTT(api_key="test-key-123")
        assert stt._api_key == "test-key-123"

    def test_reads_env_var(self) -> None:
        os.environ["ASSEMBLYAI_API_KEY"] = "env-key-456"
        try:
            stt = AssemblyAISTT()
            assert stt._api_key == "env-key-456"
        finally:
            del os.environ["ASSEMBLYAI_API_KEY"]

    def test_default_language(self) -> None:
        stt = AssemblyAISTT(api_key="key")
        assert stt._language_code == "en"

    def test_custom_language(self) -> None:
        stt = AssemblyAISTT(api_key="key", language_code="es")
        assert stt._language_code == "es"

    def test_is_stt_provider(self) -> None:
        from intent_engine.stt.base import STTProvider

        stt = AssemblyAISTT(api_key="key")
        assert isinstance(stt, STTProvider)


class TestAssemblyAISTTTranscribe:
    def test_import_error_without_sdk(self) -> None:
        import sys
        from unittest.mock import patch

        stt = AssemblyAISTT(api_key="key")
        with patch.dict(sys.modules, {"assemblyai": None}), pytest.raises(
            ImportError, match="assemblyai is required"
        ):
            asyncio.run(
                stt.transcribe("/some/audio.wav")
            )


class _FakeAssemblyAI:
    """Installs a stand-in ``assemblyai`` module with the SDK's blocking surface.

    ``transcribe`` records the thread it runs on and the API key the
    transcriber was built with, and can be slowed down or made to fail.
    """

    class TranscriptStatus(enum.Enum):
        completed = "completed"
        error = "error"

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.delay = 0.0
        self.transcript: Any = self._transcript()
        self.threads: list[threading.Thread] = []
        self.configs: list[Any] = []
        fake = self

        class Transcriber:
            def __init__(self, config: Any = None) -> None:
                time.sleep(0.05)  # a window in which another thread could change settings
                self.config = config
                self.api_key = module.settings.api_key
                fake.configs.append(config)

            def transcribe(self, path: str) -> Any:
                fake.threads.append(threading.current_thread())
                time.sleep(fake.delay)
                transcript = fake.transcript
                if transcript.status == fake.TranscriptStatus.completed:
                    return SimpleNamespace(**{**vars(transcript), "text": self.api_key})
                return transcript

        module = types.ModuleType("assemblyai")
        module.settings = SimpleNamespace(api_key=None)  # type: ignore[attr-defined]
        module.TranscriptionConfig = lambda **kw: SimpleNamespace(**kw)  # type: ignore[attr-defined]
        module.Transcriber = Transcriber  # type: ignore[attr-defined]
        module.TranscriptStatus = self.TranscriptStatus  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "assemblyai", module)
        self.module = module

    @classmethod
    def _transcript(cls, **fields: Any) -> Any:
        words = [
            SimpleNamespace(text="I", start=0, end=200, speaker=None),
            SimpleNamespace(text="need", start=200, end=400, speaker=None),
            SimpleNamespace(text="help.", start=400, end=800, speaker=None),
        ]
        body: dict[str, Any] = {
            "status": cls.TranscriptStatus.completed,
            "text": "I need help.",
            "words": words,
            "error": None,
        }
        body.update(fields)
        return SimpleNamespace(**body)


@pytest.fixture()
def fake_assemblyai(monkeypatch: pytest.MonkeyPatch) -> _FakeAssemblyAI:
    return _FakeAssemblyAI(monkeypatch)


class TestAssemblyAISTTWithFakeSDK:
    def test_blocking_call_runs_off_the_event_loop(
        self, fake_assemblyai: _FakeAssemblyAI
    ) -> None:
        fake_assemblyai.delay = 0.4

        async def scenario() -> int:
            async with Heartbeat() as heartbeat:
                _, ticks = await heartbeat.ticks_during(
                    AssemblyAISTT(api_key="key").transcribe("/some/audio.wav")
                )
            return ticks

        assert asyncio.run(scenario()) >= 5
        assert fake_assemblyai.threads
        assert all(t is not threading.main_thread() for t in fake_assemblyai.threads)

    def test_passes_language_code_to_the_sdk(self, fake_assemblyai: _FakeAssemblyAI) -> None:
        asyncio.run(AssemblyAISTT(api_key="key", language_code="es").transcribe("/a.wav"))
        assert fake_assemblyai.configs[0].language_code == "es"

    def test_concurrent_transcriptions_use_their_own_api_key(
        self, fake_assemblyai: _FakeAssemblyAI
    ) -> None:
        async def scenario() -> list[str]:
            results = await asyncio.gather(
                AssemblyAISTT(api_key="key-one").transcribe("/a.wav"),
                AssemblyAISTT(api_key="key-two").transcribe("/b.wav"),
            )
            return [r.text for r in results]

        # The fake echoes the key each transcriber was built with as the text.
        assert asyncio.run(scenario()) == ["key-one", "key-two"]

    def test_failed_transcript_is_stt_error(self, fake_assemblyai: _FakeAssemblyAI) -> None:
        fake_assemblyai.transcript = _FakeAssemblyAI._transcript(
            status=_FakeAssemblyAI.TranscriptStatus.error, error="Audio file is corrupt"
        )
        with pytest.raises(STTError, match="Audio file is corrupt"):
            asyncio.run(AssemblyAISTT(api_key="key").transcribe("/a.wav"))

    def test_sdk_exception_is_stt_error_with_the_cause(
        self, fake_assemblyai: _FakeAssemblyAI
    ) -> None:
        boom = ConnectionError("upload failed: connection reset")

        def fail(self: Any, path: str) -> Any:
            raise boom

        fake_assemblyai.module.Transcriber.transcribe = fail
        with pytest.raises(STTError, match="connection reset") as info:
            asyncio.run(AssemblyAISTT(api_key="key").transcribe("/a.wav"))
        assert info.value.__cause__ is boom

    def test_missing_audio_file_is_not_wrapped(self, fake_assemblyai: _FakeAssemblyAI) -> None:
        def missing(self: Any, path: str) -> Any:
            raise FileNotFoundError(2, "No such file or directory", path)

        fake_assemblyai.module.Transcriber.transcribe = missing
        with pytest.raises(FileNotFoundError):
            asyncio.run(AssemblyAISTT(api_key="key").transcribe("/nonexistent.wav"))

    def test_speaker_labels_and_punctuation_are_kept(
        self, fake_assemblyai: _FakeAssemblyAI
    ) -> None:
        fake_assemblyai.transcript = _FakeAssemblyAI._transcript(
            words=[
                SimpleNamespace(text="Hi.", start=0, end=300, speaker="A"),
                SimpleNamespace(text="Yes.", start=500, end=800, speaker="B"),
            ]
        )
        result = asyncio.run(AssemblyAISTT(api_key="key").transcribe("/a.wav"))
        assert [(a.word, a.speaker) for a in result.alignments] == [("Hi.", "A"), ("Yes.", "B")]

    def test_words_that_end_before_they_start_are_stt_error(
        self, fake_assemblyai: _FakeAssemblyAI
    ) -> None:
        fake_assemblyai.transcript = _FakeAssemblyAI._transcript(
            words=[SimpleNamespace(text="late", start=2000, end=1000, speaker=None)]
        )
        with pytest.raises(STTError, match="late"):
            asyncio.run(AssemblyAISTT(api_key="key").transcribe("/a.wav"))

    def test_transcript_without_words_gives_empty_alignments(
        self, fake_assemblyai: _FakeAssemblyAI
    ) -> None:
        fake_assemblyai.transcript = _FakeAssemblyAI._transcript(text=None, words=None)
        result = asyncio.run(AssemblyAISTT(api_key="key").transcribe("/a.wav"))
        assert result.alignments == []


class TestAssemblyAIResponseParsing:
    """Test that the expected output types are correct."""

    def test_word_alignment_creation(self) -> None:
        wa = WordAlignment(word="hello", start_ms=100, end_ms=500)
        assert wa.word == "hello"
        assert wa.start_ms == 100
        assert wa.end_ms == 500
