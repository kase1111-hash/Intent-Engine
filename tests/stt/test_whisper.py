"""Tests for the Whisper STT adapter."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from prosody_protocol import IMLAssembler, WordAlignment

from intent_engine.errors import STTError
from intent_engine.stt.base import TranscriptionResult
from intent_engine.stt.whisper import WhisperSTT

from .helpers import Heartbeat


def _make_whisper_result() -> dict:
    """Create a mock Whisper transcription result."""
    return {
        "text": " Hello world, how are you?",
        "language": "en",
        "segments": [
            {
                "id": 0,
                "text": " Hello world, how are you?",
                "words": [
                    {"word": " Hello", "start": 0.0, "end": 0.5},
                    {"word": " world,", "start": 0.5, "end": 1.0},
                    {"word": " how", "start": 1.2, "end": 1.5},
                    {"word": " are", "start": 1.5, "end": 1.7},
                    {"word": " you?", "start": 1.7, "end": 2.0},
                ],
            }
        ],
    }


def _install_mock_whisper() -> MagicMock:
    """Install a mock 'whisper' module into sys.modules and return it."""
    mock_whisper = types.ModuleType("whisper")
    mock_whisper.transcribe = MagicMock()  # type: ignore[attr-defined]
    mock_whisper.load_model = MagicMock()  # type: ignore[attr-defined]
    sys.modules["whisper"] = mock_whisper
    return mock_whisper  # type: ignore[return-value]


def _remove_mock_whisper() -> None:
    sys.modules.pop("whisper", None)


class TestWhisperSTTConstruction:
    def test_default_params(self) -> None:
        stt = WhisperSTT()
        assert stt._model_size == "base"
        assert stt._device == "cpu"
        assert stt._language is None
        assert stt._model is None

    def test_custom_params(self) -> None:
        stt = WhisperSTT(model_size="large-v3", device="cuda", language="en")
        assert stt._model_size == "large-v3"
        assert stt._device == "cuda"
        assert stt._language == "en"

    def test_accepts_kwargs(self) -> None:
        stt = WhisperSTT(extra_param="ignored")
        assert stt._model_size == "base"


class TestWhisperSTTTranscribe:
    def test_transcribe_returns_transcription_result(self) -> None:
        mock_whisper = _install_mock_whisper()
        try:
            mock_whisper.transcribe.return_value = _make_whisper_result()

            stt = WhisperSTT()
            stt._model = MagicMock()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"fake audio data")
                audio_path = f.name

            result = asyncio.run(
                stt.transcribe(audio_path)
            )
            Path(audio_path).unlink()

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello world, how are you?"
            assert result.language == "en"
        finally:
            _remove_mock_whisper()

    def test_word_alignments_are_correct(self) -> None:
        mock_whisper = _install_mock_whisper()
        try:
            mock_whisper.transcribe.return_value = _make_whisper_result()

            stt = WhisperSTT()
            stt._model = MagicMock()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"fake audio data")
                audio_path = f.name

            result = asyncio.run(
                stt.transcribe(audio_path)
            )
            Path(audio_path).unlink()

            assert len(result.alignments) == 5
            assert all(isinstance(a, WordAlignment) for a in result.alignments)

            # Check first word
            assert result.alignments[0].word == "Hello"
            assert result.alignments[0].start_ms == 0
            assert result.alignments[0].end_ms == 500

            # Check last word
            assert result.alignments[4].word == "you?"
            assert result.alignments[4].start_ms == 1700
            assert result.alignments[4].end_ms == 2000
        finally:
            _remove_mock_whisper()

    def test_start_ms_before_end_ms(self) -> None:
        mock_whisper = _install_mock_whisper()
        try:
            mock_whisper.transcribe.return_value = _make_whisper_result()

            stt = WhisperSTT()
            stt._model = MagicMock()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"fake audio data")
                audio_path = f.name

            result = asyncio.run(
                stt.transcribe(audio_path)
            )
            Path(audio_path).unlink()

            for alignment in result.alignments:
                assert alignment.start_ms <= alignment.end_ms
        finally:
            _remove_mock_whisper()

    def test_file_not_found_raises(self) -> None:
        stt = WhisperSTT()
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            asyncio.run(
                stt.transcribe("/nonexistent/audio.wav")
            )

    def test_empty_segments(self) -> None:
        mock_whisper = _install_mock_whisper()
        try:
            mock_whisper.transcribe.return_value = {
                "text": "",
                "language": "en",
                "segments": [],
            }

            stt = WhisperSTT()
            stt._model = MagicMock()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"fake audio data")
                audio_path = f.name

            result = asyncio.run(
                stt.transcribe(audio_path)
            )
            Path(audio_path).unlink()

            assert result.text == ""
            assert result.alignments == []
        finally:
            _remove_mock_whisper()

    def test_skips_empty_words(self) -> None:
        mock_whisper = _install_mock_whisper()
        try:
            mock_whisper.transcribe.return_value = {
                "text": " Hello",
                "language": "en",
                "segments": [
                    {
                        "words": [
                            {"word": " Hello", "start": 0.0, "end": 0.5},
                            {"word": "  ", "start": 0.5, "end": 0.6},
                            {"word": "", "start": 0.6, "end": 0.7},
                        ],
                    }
                ],
            }

            stt = WhisperSTT()
            stt._model = MagicMock()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"fake audio data")
                audio_path = f.name

            result = asyncio.run(
                stt.transcribe(audio_path)
            )
            Path(audio_path).unlink()

            assert len(result.alignments) == 1
            assert result.alignments[0].word == "Hello"
        finally:
            _remove_mock_whisper()


class TestWhisperSTTLazyLoad:
    def test_import_error_without_whisper(self) -> None:
        stt = WhisperSTT()
        with patch.dict("sys.modules", {"whisper": None}), pytest.raises(
            ImportError, match="openai-whisper is required"
        ):
            stt._load_model()


@pytest.fixture()
def whisper_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A stand-in ``whisper`` module whose ``transcribe`` returns a canned result."""
    module = types.ModuleType("whisper")
    module.load_model = MagicMock(return_value=MagicMock(name="model"))  # type: ignore[attr-defined]
    module.transcribe = MagicMock(return_value=_make_whisper_result())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "whisper", module)
    return module


@pytest.fixture()
def audio_file(tmp_path: Path) -> str:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"fake audio data")
    return str(path)


class TestWhisperSTTEventLoop:
    def test_model_load_and_transcription_run_off_the_event_loop(
        self, whisper_module: Any, audio_file: str
    ) -> None:
        threads: list[threading.Thread] = []

        def slow_load(*args: Any, **kwargs: Any) -> Any:
            threads.append(threading.current_thread())
            time.sleep(0.3)
            return MagicMock()

        def slow_transcribe(*args: Any, **kwargs: Any) -> Any:
            threads.append(threading.current_thread())
            time.sleep(0.3)
            return _make_whisper_result()

        whisper_module.load_model.side_effect = slow_load
        whisper_module.transcribe.side_effect = slow_transcribe

        async def scenario() -> int:
            async with Heartbeat() as heartbeat:
                _, ticks = await heartbeat.ticks_during(WhisperSTT().transcribe(audio_file))
            return ticks

        # A free loop ticks ~30 times in 0.6 s; a blocked one not at all.
        assert asyncio.run(scenario()) >= 10
        assert len(threads) == 2
        assert all(t is not threading.main_thread() for t in threads)

    def test_concurrent_transcriptions_share_one_model_load_and_do_not_overlap(
        self, whisper_module: Any, audio_file: str
    ) -> None:
        # One Whisper model is not safe for overlapping decodes (kv-cache hooks are
        # installed on the shared modules), so calls must be serialized.
        running = 0
        peak = 0
        guard = threading.Lock()

        def transcribe(*args: Any, **kwargs: Any) -> Any:
            nonlocal running, peak
            with guard:
                running += 1
                peak = max(peak, running)
            time.sleep(0.05)
            with guard:
                running -= 1
            return _make_whisper_result()

        def load(*args: Any, **kwargs: Any) -> Any:
            time.sleep(0.05)
            return MagicMock()

        whisper_module.transcribe.side_effect = transcribe
        whisper_module.load_model.side_effect = load

        async def scenario() -> None:
            stt = WhisperSTT()
            await asyncio.gather(*(stt.transcribe(audio_file) for _ in range(4)))

        asyncio.run(scenario())
        assert peak == 1
        assert whisper_module.load_model.call_count == 1
        assert whisper_module.transcribe.call_count == 4


class TestWhisperSTTContract:
    def test_calls_whisper_with_word_timestamps_and_language(
        self, whisper_module: Any, audio_file: str
    ) -> None:
        model = MagicMock()
        whisper_module.load_model.return_value = model
        stt = WhisperSTT(model_size="tiny", device="cuda", language="en")
        asyncio.run(stt.transcribe(audio_file))
        whisper_module.load_model.assert_called_once_with("tiny", device="cuda")
        whisper_module.transcribe.assert_called_once_with(
            model, audio_file, language="en", word_timestamps=True
        )

    def test_sentence_punctuation_reaches_iml(self, whisper_module: Any, audio_file: str) -> None:
        whisper_module.transcribe.return_value = {
            "text": " I need help. Cancel my order please.",
            "language": "en",
            "segments": [
                {
                    "words": [
                        {"word": " I", "start": 0.0, "end": 0.2},
                        {"word": " need", "start": 0.2, "end": 0.4},
                        {"word": " help.", "start": 0.4, "end": 0.8},
                        {"word": " Cancel", "start": 1.0, "end": 1.3},
                        {"word": " my", "start": 1.3, "end": 1.4},
                        {"word": " order", "start": 1.4, "end": 1.7},
                        {"word": " please.", "start": 1.7, "end": 2.1},
                    ]
                }
            ],
        }
        result = asyncio.run(WhisperSTT().transcribe(audio_file))
        doc = IMLAssembler().assemble(result.alignments, [], [], language=result.language)
        assert len(doc.utterances) == 2

    def test_accepts_the_result_shape_openai_whisper_returns(
        self, whisper_module: Any, audio_file: str
    ) -> None:
        # As built by whisper.transcribe(word_timestamps=True): extra segment keys,
        # per-word probability, and an emptied segment that keeps ``words == []``.
        whisper_module.transcribe.return_value = {
            "text": " Hello there.",
            "language": "en",
            "segments": [
                {
                    "id": 0, "seek": 0, "start": 0.0, "end": 1.2,
                    "text": " Hello there.", "tokens": [50364, 2425, 456, 13, 50424],
                    "temperature": 0.0, "avg_logprob": -0.2,
                    "compression_ratio": 0.8, "no_speech_prob": 0.01,
                    "words": [
                        {"word": " Hello", "start": 0.0, "end": 0.6, "probability": 0.93},
                        {"word": " there.", "start": 0.6, "end": 1.2, "probability": 0.88},
                    ],
                },
                {
                    "id": 1, "seek": 3000, "start": 30.0, "end": 30.0, "text": "",
                    "tokens": [], "temperature": 0.0, "avg_logprob": -0.9,
                    "compression_ratio": 0.0, "no_speech_prob": 0.7, "words": [],
                },
            ],
        }
        result = asyncio.run(WhisperSTT().transcribe(audio_file))
        assert result.text == "Hello there."
        assert [(a.word, a.start_ms, a.end_ms) for a in result.alignments] == [
            ("Hello", 0, 600),
            ("there.", 600, 1200),
        ]

    def test_times_are_rounded_to_whole_milliseconds(
        self, whisper_module: Any, audio_file: str
    ) -> None:
        # 1.005 * 1000 is 1004.9999999999999 in floating point; int() would give 1004.
        whisper_module.transcribe.return_value = {
            "text": "ok",
            "language": "en",
            "segments": [{"words": [{"word": " ok", "start": 1.001, "end": 1.005}]}],
        }
        result = asyncio.run(WhisperSTT().transcribe(audio_file))
        assert (result.alignments[0].start_ms, result.alignments[0].end_ms) == (1001, 1005)

    def test_words_that_end_before_they_start_are_stt_error(
        self, whisper_module: Any, audio_file: str
    ) -> None:
        whisper_module.transcribe.return_value = {
            "text": "late",
            "language": "en",
            "segments": [{"words": [{"word": " late", "start": 2.0, "end": 1.0}]}],
        }
        with pytest.raises(STTError, match="late"):
            asyncio.run(WhisperSTT().transcribe(audio_file))

    def test_segments_without_word_timestamps_are_stt_error(
        self, whisper_module: Any, audio_file: str
    ) -> None:
        whisper_module.transcribe.return_value = {
            "text": "hi",
            "language": "en",
            "segments": [{"text": " hi"}],
        }
        with pytest.raises(STTError, match="word timestamps"):
            asyncio.run(WhisperSTT().transcribe(audio_file))

    def test_whisper_failure_is_stt_error_with_the_real_cause(
        self, whisper_module: Any, audio_file: str
    ) -> None:
        boom = RuntimeError("Failed to load audio: [Errno 2] No such file or directory: 'ffmpeg'")
        whisper_module.transcribe.side_effect = boom
        with pytest.raises(STTError, match="Failed to load audio") as info:
            asyncio.run(WhisperSTT().transcribe(audio_file))
        assert info.value.__cause__ is boom

    def test_missing_package_is_still_an_import_error(self, audio_file: str) -> None:
        with patch.dict("sys.modules", {"whisper": None}), pytest.raises(
            ImportError, match="openai-whisper is required"
        ):
            asyncio.run(WhisperSTT().transcribe(audio_file))
