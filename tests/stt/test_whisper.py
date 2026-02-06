"""Tests for the Whisper STT adapter."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from prosody_protocol import WordAlignment

from intent_engine.stt.base import TranscriptionResult
from intent_engine.stt.whisper import WhisperSTT


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

            result = asyncio.get_event_loop().run_until_complete(
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

            result = asyncio.get_event_loop().run_until_complete(
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

            result = asyncio.get_event_loop().run_until_complete(
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
            asyncio.get_event_loop().run_until_complete(
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

            result = asyncio.get_event_loop().run_until_complete(
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

            result = asyncio.get_event_loop().run_until_complete(
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
