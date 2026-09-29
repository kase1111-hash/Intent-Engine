"""Prosody analysis failures degrade to text-only IML.

Upstream reports every read/analysis failure as ``AudioProcessingError``
(a ``ProsodyProtocolError``): audio that is too short, sampled too low, not
decodable, empty.  The STT result is already paid for, so the turn goes on
with the words and no prosody instead of failing.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from prosody_protocol import AudioProcessingError, ConversionError, ProsodyProtocolError

from intent_engine.engine import IntentEngine
from intent_engine.models.result import Result
from intent_engine.stt.base import TranscriptionResult
from tests.conftest import assert_valid_iml, create_mocked_engine, make_flat_speech

TRANSCRIPT = "I am fine thank you today."


def _engine(**kwargs) -> IntentEngine:
    engine = create_mocked_engine(**kwargs)
    alignments, _ = make_flat_speech()
    engine._stt.transcribe = AsyncMock(
        return_value=TranscriptionResult(text=TRANSCRIPT, alignments=alignments, language="en")
    )
    return engine


def _write_wav(path: Path, samples: int, rate: int = 16000) -> str:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<h", 0) * samples)
    return str(path)


def _process(engine: IntentEngine, path: str) -> Result:
    return asyncio.run(engine.process_voice_input(path, use_cache=False))


def _assert_text_only(result: Result) -> None:
    assert result.text == TRANSCRIPT
    assert result.prosody_features == []
    assert (result.emotion, result.confidence, result.suggested_tone) == ("neutral", 0.0, "neutral")
    # the LLM still gets the words, as valid IML without prosody
    assert TRANSCRIPT in result.iml
    assert "<prosody" not in result.iml
    assert_valid_iml(result.iml)


class TestRealAnalyzerFailures:
    """The real ProsodyAnalyzer on audio it cannot analyse."""

    @pytest.fixture(autouse=True)
    def _need_audio_stack(self) -> None:
        pytest.importorskip("numpy")
        pytest.importorskip("parselmouth")

    def test_audio_too_short(self, tmp_path: Path) -> None:
        path = _write_wav(tmp_path / "short.wav", samples=800)  # 50 ms

        _assert_text_only(_process(_engine(), path))

    def test_sample_rate_too_low(self, tmp_path: Path) -> None:
        path = _write_wav(tmp_path / "low.wav", samples=4000, rate=2000)

        _assert_text_only(_process(_engine(), path))

    def test_not_audio_at_all(self, tmp_path: Path) -> None:
        path = tmp_path / "voice.ogg"  # e.g. a voice note the decoder cannot read
        path.write_bytes(b"OggS" + b"\0" * 100)

        _assert_text_only(_process(_engine(), str(path)))

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.wav"
        path.write_bytes(b"")

        _assert_text_only(_process(_engine(), str(path)))


class TestInjectedFailures:
    @pytest.mark.parametrize(
        "error",
        [
            AudioProcessingError("cannot read"),
            ProsodyProtocolError("other upstream failure"),
            RuntimeError("praat"),
            OSError("disk"),
            ValueError("bad span"),
        ],
        ids=lambda e: type(e).__name__,
    )
    def test_analysis_failure_falls_back(self, tmp_path: Path, error: Exception) -> None:
        engine = _engine()
        engine._analyzer.analyze = MagicMock(side_effect=error)
        path = _write_wav(tmp_path / "a.wav", samples=16000)

        _assert_text_only(_process(engine, path))

    def test_pause_detection_failure_drops_prosody_altogether(self, tmp_path: Path) -> None:
        engine = _engine()
        _, features = make_flat_speech()
        engine._analyzer.analyze = MagicMock(return_value=features)
        engine._analyzer.detect_pauses = MagicMock(side_effect=AudioProcessingError("no"))
        path = _write_wav(tmp_path / "a.wav", samples=16000)

        _assert_text_only(_process(engine, path))

    def test_programming_errors_are_not_swallowed(self, tmp_path: Path) -> None:
        engine = _engine()
        engine._analyzer.analyze = MagicMock(side_effect=TypeError("bug"))
        path = _write_wav(tmp_path / "a.wav", samples=16000)

        with pytest.raises(TypeError, match="bug"):
            _process(engine, path)

    def test_failure_after_analysis_is_not_hidden_as_a_fallback(self, tmp_path: Path) -> None:
        engine = _engine()
        _, features = make_flat_speech()
        engine._analyzer.analyze = MagicMock(return_value=features)
        engine._analyzer.detect_pauses = MagicMock(return_value=[])
        engine._parser.to_iml_string = MagicMock(side_effect=ConversionError("bad text"))
        path = _write_wav(tmp_path / "a.wav", samples=16000)

        with pytest.raises(ConversionError):
            _process(engine, path)

    def test_fallback_result_is_not_cached(self, tmp_path: Path) -> None:
        # a degraded result must not stick if the failure was transient
        engine = _engine()
        engine._analyzer.analyze = MagicMock(side_effect=AudioProcessingError("no"))
        path = _write_wav(tmp_path / "a.wav", samples=16000)

        asyncio.run(engine.process_voice_input(path))
        assert len(engine._cache) == 0
        engine._analyzer.analyze = MagicMock(return_value=make_flat_speech()[1])
        engine._analyzer.detect_pauses = MagicMock(return_value=[])
        recovered = asyncio.run(engine.process_voice_input(path))

        assert engine._stt.transcribe.call_count == 2
        assert len(recovered.prosody_features) > 0


class TestFallbackLogging:
    def test_warning_names_the_failure_without_transcript_or_traceback(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine = _engine()
        engine._analyzer.analyze = MagicMock(
            side_effect=AudioProcessingError("cannot read /uploads/jane-doe/turn.wav")
        )
        path = _write_wav(tmp_path / "a.wav", samples=16000)

        with caplog.at_level(logging.DEBUG, logger="intent_engine"):
            _process(engine, path)

        loud = [r for r in caplog.records if r.levelno >= logging.INFO]
        warnings = [r for r in loud if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "AudioProcessingError" in warnings[0].getMessage()
        for record in loud:
            assert record.exc_info is None
            text = record.getMessage()
            assert "fine" not in text and "jane-doe" not in text
        # the details are there for whoever turns on debug logging
        assert any(r.levelno == logging.DEBUG and r.exc_info for r in caplog.records)
