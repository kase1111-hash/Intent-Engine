"""STT text without word timings still reaches the IML.

Some providers return a transcript but no per-word timings.  The assembler
builds the document from word alignments, so the transcript used to vanish:
the LLM was handed an empty ``<utterance>``.  The engine now measures the
whole recording as one span holding the text (upstream's
``ProsodyAnalyzer.analyze_recording``), like upstream's own handling of a
transcript without timings.
"""

from __future__ import annotations

import asyncio
import struct
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from prosody_protocol import AudioProcessingError, SpanFeatures

from intent_engine.engine import IntentEngine
from intent_engine.models.result import Result
from intent_engine.stt.base import TranscriptionResult
from tests.conftest import assert_valid_iml, create_mocked_engine

TEXT = "I really mean it"


def _engine(text: str = TEXT, **kwargs) -> IntentEngine:
    engine = create_mocked_engine(**kwargs)
    engine._stt.transcribe = AsyncMock(
        return_value=TranscriptionResult(text=text, alignments=[], language="en")
    )
    return engine


def _write_wav(path: Path, samples: int = 16000) -> str:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<h", 0) * samples)
    return str(path)


def _process(engine: IntentEngine, path: str) -> Result:
    return asyncio.run(engine.process_voice_input(path, use_cache=False))


class TestWithMockedAnalyzer:
    def test_the_text_is_in_the_iml(self, tmp_path: Path) -> None:
        engine = _engine()
        whole = SpanFeatures(start_ms=0, end_ms=2500, text=TEXT, f0_mean=150.0)
        engine._analyzer.analyze_recording = MagicMock(return_value=whole)
        engine._analyzer.detect_pauses = MagicMock(return_value=[])

        result = _process(engine, _write_wav(tmp_path / "a.wav"))

        assert TEXT in result.iml
        assert_valid_iml(result.iml)
        assert result.text == TEXT

    def test_the_whole_recording_is_measured_as_one_span(self, tmp_path: Path) -> None:
        engine = _engine("  I  really\nmean it ")
        whole = SpanFeatures(start_ms=0, end_ms=2500, text=TEXT, f0_mean=150.0)
        engine._analyzer.analyze_recording = MagicMock(return_value=whole)
        engine._analyzer.analyze = MagicMock(side_effect=AssertionError("per-word analysis"))
        engine._analyzer.detect_pauses = MagicMock(return_value=[])
        path = _write_wav(tmp_path / "a.wav")

        result = _process(engine, path)

        engine._analyzer.analyze_recording.assert_called_once_with(path, text=TEXT)
        assert result.prosody_features == [whole]
        assert TEXT in result.iml  # whitespace normalized

    def test_timed_words_are_still_analysed_per_word(self, tmp_path: Path) -> None:
        from prosody_protocol import WordAlignment

        engine = _engine()
        engine._stt.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text="Hello world", alignments=[WordAlignment("Hello", 0, 400)], language="en"
            )
        )
        engine._analyzer.analyze = MagicMock(return_value=[])
        engine._analyzer.analyze_recording = MagicMock(side_effect=AssertionError("whole"))
        engine._analyzer.detect_pauses = MagicMock(return_value=[])

        _process(engine, _write_wav(tmp_path / "a.wav"))

        engine._analyzer.analyze.assert_called_once()

    def test_silence_with_no_text_stays_an_empty_utterance(self, tmp_path: Path) -> None:
        engine = _engine("")
        engine._analyzer.analyze = MagicMock(return_value=[])
        engine._analyzer.analyze_recording = MagicMock(side_effect=AssertionError("whole"))
        engine._analyzer.detect_pauses = MagicMock(return_value=[])

        result = _process(engine, _write_wav(tmp_path / "a.wav"))

        assert result.text == ""
        assert_valid_iml(result.iml)

    def test_the_text_survives_a_failed_analysis(self, tmp_path: Path) -> None:
        engine = _engine()
        engine._analyzer.analyze_recording = MagicMock(side_effect=AudioProcessingError("no"))

        result = _process(engine, _write_wav(tmp_path / "a.wav"))

        assert TEXT in result.iml
        assert_valid_iml(result.iml)
        assert result.prosody_features == []
        assert (result.emotion, result.confidence) == ("neutral", 0.0)

    def test_a_multi_sentence_transcript_is_one_valid_utterance(self, tmp_path: Path) -> None:
        text = "Hello there. How are you? I am fine."
        engine = _engine(text)
        engine._analyzer.analyze_recording = MagicMock(
            return_value=SpanFeatures(start_ms=0, end_ms=4000, text=text)
        )
        engine._analyzer.detect_pauses = MagicMock(return_value=[])

        result = _process(engine, _write_wav(tmp_path / "a.wav"))

        assert text in result.iml
        assert_valid_iml(result.iml)


class TestOnRealAudio:
    @pytest.fixture(autouse=True)
    def _need_audio_stack(self) -> None:
        pytest.importorskip("numpy")
        pytest.importorskip("parselmouth")

    def test_the_llm_sees_the_words_and_the_profile_still_applies(self, tmp_path: Path) -> None:
        from prosody_protocol import ProsodyMapping, ProsodyProfile

        from tests.synth_audio import NEUTRAL, write_recording

        wav = tmp_path / "flat.wav"
        write_recording(wav, [NEUTRAL])  # one monotone sentence
        engine = _engine()
        engine.set_profile(
            ProsodyProfile(
                "1.0.0", "u", None, (ProsodyMapping({"pitch_contour": "flat"}, "calm", 0.6),)
            )
        )

        result = _process(engine, str(wav))

        assert TEXT in result.iml
        assert 'x-profile="pitch_contour=flat"' in result.iml
        assert (result.emotion, result.confidence) == ("calm", 0.6)
        assert len(result.prosody_features) == 1
        assert result.prosody_features[0].end_ms > 3000  # the whole recording
        assert_valid_iml(result.iml)
