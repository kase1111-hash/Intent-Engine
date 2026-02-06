"""End-to-end integration tests for IntentEngine (mocked providers)."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prosody_protocol import (
    IMLDocument,
    PauseInterval,
    Segment,
    SpanFeatures,
    Utterance,
    ValidationResult,
    WordAlignment,
)

from intent_engine.engine import IntentEngine
from intent_engine.errors import IntentEngineError, LLMError, STTError, TTSError
from intent_engine.llm.base import InterpretationResult
from intent_engine.models.audio import Audio
from intent_engine.models.decision import Decision
from intent_engine.models.response import Response
from intent_engine.models.result import Result
from intent_engine.stt.base import TranscriptionResult
from intent_engine.tts.base import SynthesisResult


# -- Fixtures and helpers --


def _make_alignment(word: str, start: int, end: int) -> WordAlignment:
    return WordAlignment(word=word, start_ms=start, end_ms=end)


def _make_features(
    text: str = "hello",
    f0_mean: float = 150.0,
    speech_rate: float = 4.0,
) -> SpanFeatures:
    return SpanFeatures(
        start_ms=0,
        end_ms=1000,
        text=text,
        f0_mean=f0_mean,
        speech_rate=speech_rate,
    )


def _make_iml_doc() -> IMLDocument:
    return IMLDocument(
        utterances=(Utterance(children=(Segment(),)),),
        version="1.0",
    )


def _create_engine(
    stt_mock: MagicMock | None = None,
    llm_mock: MagicMock | None = None,
    tts_mock: MagicMock | None = None,
    constitutional_rules: str | None = None,
) -> IntentEngine:
    """Create an IntentEngine with mocked providers."""
    with patch("intent_engine.engine.create_stt_provider") as mock_stt_factory, \
         patch("intent_engine.engine.create_llm_provider") as mock_llm_factory, \
         patch("intent_engine.engine.create_tts_provider") as mock_tts_factory:

        mock_stt_factory.return_value = stt_mock or MagicMock()
        mock_llm_factory.return_value = llm_mock or MagicMock()
        mock_tts_factory.return_value = tts_mock or MagicMock()

        engine = IntentEngine(
            stt_provider="whisper-prosody",
            llm_provider="claude",
            tts_provider="coqui",
            constitutional_rules=constitutional_rules,
            stt_kwargs={},
            llm_kwargs={"api_key": "fake"},
            tts_kwargs={},
        )
    return engine


class TestIntentEngineConstruction:
    def test_creates_with_mocked_providers(self) -> None:
        engine = _create_engine()
        assert engine is not None

    def test_has_prosody_protocol_components(self) -> None:
        engine = _create_engine()
        assert engine._analyzer is not None
        assert engine._assembler is not None
        assert engine._parser is not None
        assert engine._validator is not None
        assert engine._emotion_classifier is not None

    def test_no_filter_by_default(self) -> None:
        engine = _create_engine()
        assert engine._filter is None

    def test_filter_loaded_from_yaml(self) -> None:
        rules_path = str(
            Path(__file__).parent / "constitutional" / "sample_rules.yaml"
        )
        engine = _create_engine(constitutional_rules=rules_path)
        assert engine._filter is not None

    def test_cache_initialized(self) -> None:
        engine = _create_engine()
        assert engine._cache == {}
        assert len(engine._cache) == 0


class TestProcessVoiceInput:
    def test_full_pipeline(self) -> None:
        stt_mock = MagicMock()
        alignments = [
            _make_alignment("Hello", 0, 500),
            _make_alignment("world", 500, 1000),
        ]
        stt_mock.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text="Hello world",
                alignments=alignments,
                language="en",
            )
        )

        engine = _create_engine(stt_mock=stt_mock)

        # Mock prosody protocol components
        features = [_make_features("Hello world")]
        engine._analyzer.analyze = MagicMock(return_value=features)
        engine._analyzer.detect_pauses = MagicMock(return_value=[])

        iml_doc = _make_iml_doc()
        engine._assembler.assemble = MagicMock(return_value=iml_doc)

        iml_string = '<iml><utterance>Hello world</utterance></iml>'
        engine._parser.to_iml_string = MagicMock(return_value=iml_string)

        engine._validator.validate = MagicMock(
            return_value=ValidationResult(valid=True)
        )

        engine._emotion_classifier.classify = MagicMock(
            return_value=("joyful", 0.85)
        )

        # Write a temp audio file for hashing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake audio")
            audio_path = f.name

        try:
            result = asyncio.run(
                engine.process_voice_input(audio_path)
            )

            assert isinstance(result, Result)
            assert result.text == "Hello world"
            assert result.emotion == "joyful"
            assert result.confidence == 0.85
            assert result.iml == iml_string
            assert result.iml_document == iml_doc
            assert result.suggested_tone == "joyful"
            assert result.prosody_features == features
        finally:
            Path(audio_path).unlink()

    def test_prosody_failure_falls_back(self) -> None:
        stt_mock = MagicMock()
        stt_mock.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text="Test", alignments=[], language="en"
            )
        )

        engine = _create_engine(stt_mock=stt_mock)

        # Prosody analysis raises
        engine._analyzer.analyze = MagicMock(side_effect=RuntimeError("no audio"))
        engine._assembler.assemble = MagicMock(return_value=_make_iml_doc())
        engine._parser.to_iml_string = MagicMock(return_value="<iml/>")
        engine._validator.validate = MagicMock(
            return_value=ValidationResult(valid=True)
        )
        engine._emotion_classifier.classify = MagicMock(
            return_value=("neutral", 0.0)
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake")
            audio_path = f.name

        try:
            result = asyncio.run(
                engine.process_voice_input(audio_path)
            )
            # Should still return a result (text-only mode)
            assert result.text == "Test"
            assert result.prosody_features == []
        finally:
            Path(audio_path).unlink()

    def test_stt_failure_raises_stt_error(self) -> None:
        stt_mock = MagicMock()
        stt_mock.transcribe = AsyncMock(side_effect=RuntimeError("API down"))

        engine = _create_engine(stt_mock=stt_mock)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake")
            audio_path = f.name

        try:
            with pytest.raises(STTError, match="STT transcription failed"):
                asyncio.run(
                    engine.process_voice_input(audio_path)
                )
        finally:
            Path(audio_path).unlink()

    def test_low_confidence_suggests_neutral_tone(self) -> None:
        stt_mock = MagicMock()
        stt_mock.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="Hi", alignments=[], language="en")
        )

        engine = _create_engine(stt_mock=stt_mock)
        engine._analyzer.analyze = MagicMock(return_value=[])
        engine._analyzer.detect_pauses = MagicMock(return_value=[])
        engine._assembler.assemble = MagicMock(return_value=_make_iml_doc())
        engine._parser.to_iml_string = MagicMock(return_value="<iml/>")
        engine._validator.validate = MagicMock(
            return_value=ValidationResult(valid=True)
        )
        engine._emotion_classifier.classify = MagicMock(
            return_value=("sarcastic", 0.3)  # low confidence
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake")
            audio_path = f.name

        try:
            result = asyncio.run(
                engine.process_voice_input(audio_path)
            )
            assert result.emotion == "sarcastic"
            assert result.suggested_tone == "neutral"  # low confidence
        finally:
            Path(audio_path).unlink()


class TestCaching:
    def test_cache_hit(self) -> None:
        stt_mock = MagicMock()
        stt_mock.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="Hi", alignments=[], language="en")
        )

        engine = _create_engine(stt_mock=stt_mock)
        engine._analyzer.analyze = MagicMock(return_value=[])
        engine._analyzer.detect_pauses = MagicMock(return_value=[])
        engine._assembler.assemble = MagicMock(return_value=_make_iml_doc())
        engine._parser.to_iml_string = MagicMock(return_value="<iml/>")
        engine._validator.validate = MagicMock(
            return_value=ValidationResult(valid=True)
        )
        engine._emotion_classifier.classify = MagicMock(
            return_value=("neutral", 0.5)
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake audio data")
            audio_path = f.name

        try:
            # First call
            result1 = asyncio.run(
                engine.process_voice_input(audio_path)
            )
            # Second call should use cache
            result2 = asyncio.run(
                engine.process_voice_input(audio_path)
            )

            assert result1 is result2
            # STT should only be called once
            assert stt_mock.transcribe.call_count == 1
        finally:
            Path(audio_path).unlink()

    def test_cache_disabled(self) -> None:
        stt_mock = MagicMock()
        stt_mock.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="Hi", alignments=[], language="en")
        )

        engine = _create_engine(stt_mock=stt_mock)
        engine._analyzer.analyze = MagicMock(return_value=[])
        engine._analyzer.detect_pauses = MagicMock(return_value=[])
        engine._assembler.assemble = MagicMock(return_value=_make_iml_doc())
        engine._parser.to_iml_string = MagicMock(return_value="<iml/>")
        engine._validator.validate = MagicMock(
            return_value=ValidationResult(valid=True)
        )
        engine._emotion_classifier.classify = MagicMock(
            return_value=("neutral", 0.5)
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake")
            audio_path = f.name

        try:
            asyncio.run(
                engine.process_voice_input(audio_path, use_cache=False)
            )
            asyncio.run(
                engine.process_voice_input(audio_path, use_cache=False)
            )

            # STT called twice when cache disabled
            assert stt_mock.transcribe.call_count == 2
        finally:
            Path(audio_path).unlink()

    def test_cache_eviction(self) -> None:
        engine = _create_engine()
        engine._cache_size = 2

        # Manually populate cache
        r1 = MagicMock(spec=Result)
        r2 = MagicMock(spec=Result)
        r3 = MagicMock(spec=Result)

        engine._cache_put("a", r1)
        engine._cache_put("b", r2)
        assert len(engine._cache) == 2

        engine._cache_put("c", r3)
        assert len(engine._cache) == 2
        assert "a" not in engine._cache  # oldest evicted
        assert "b" in engine._cache
        assert "c" in engine._cache


class TestGenerateResponse:
    def test_returns_response(self) -> None:
        llm_mock = MagicMock()
        llm_mock.interpret = AsyncMock(
            return_value=InterpretationResult(
                intent="greet",
                response_text="Hello! How can I help?",
                suggested_emotion="joyful",
            )
        )

        engine = _create_engine(llm_mock=llm_mock)

        result = asyncio.run(
            engine.generate_response("<utterance>Hello</utterance>")
        )

        assert isinstance(result, Response)
        assert result.text == "Hello! How can I help?"
        assert result.emotion == "joyful"

    def test_passes_context(self) -> None:
        llm_mock = MagicMock()
        llm_mock.interpret = AsyncMock(
            return_value=InterpretationResult(
                intent="test", response_text="ok", suggested_emotion="neutral"
            )
        )

        engine = _create_engine(llm_mock=llm_mock)

        asyncio.run(
            engine.generate_response("<iml/>", context="Support scenario")
        )

        call_kwargs = llm_mock.interpret.call_args
        assert "Support scenario" in call_kwargs.kwargs.get("context", "")

    def test_passes_tone_hint(self) -> None:
        llm_mock = MagicMock()
        llm_mock.interpret = AsyncMock(
            return_value=InterpretationResult(
                intent="test", response_text="ok", suggested_emotion="neutral"
            )
        )

        engine = _create_engine(llm_mock=llm_mock)

        asyncio.run(
            engine.generate_response("<iml/>", tone="empathetic")
        )

        call_kwargs = llm_mock.interpret.call_args
        assert "empathetic" in call_kwargs.kwargs.get("context", "")

    def test_llm_failure_raises_llm_error(self) -> None:
        llm_mock = MagicMock()
        llm_mock.interpret = AsyncMock(side_effect=RuntimeError("API error"))

        engine = _create_engine(llm_mock=llm_mock)

        with pytest.raises(LLMError, match="LLM interpretation failed"):
            asyncio.run(
                engine.generate_response("<iml/>")
            )


class TestSynthesizeSpeech:
    def test_returns_audio(self) -> None:
        tts_mock = MagicMock()
        tts_mock.synthesize = AsyncMock(
            return_value=SynthesisResult(
                audio_data=b"fake audio bytes",
                format="wav",
                sample_rate=22050,
                duration=1.5,
            )
        )

        engine = _create_engine(tts_mock=tts_mock)

        result = asyncio.run(
            engine.synthesize_speech("Hello world", emotion="joyful")
        )

        assert isinstance(result, Audio)
        assert result.data == b"fake audio bytes"
        assert result.format == "wav"
        assert result.sample_rate == 22050
        assert result.duration == 1.5

    def test_passes_emotion(self) -> None:
        tts_mock = MagicMock()
        tts_mock.synthesize = AsyncMock(
            return_value=SynthesisResult(audio_data=b"data")
        )

        engine = _create_engine(tts_mock=tts_mock)

        asyncio.run(
            engine.synthesize_speech("Test", emotion="sad")
        )

        call_kwargs = tts_mock.synthesize.call_args
        assert call_kwargs.kwargs.get("emotion") == "sad"

    def test_tts_failure_raises_tts_error(self) -> None:
        tts_mock = MagicMock()
        tts_mock.synthesize = AsyncMock(side_effect=RuntimeError("TTS down"))

        engine = _create_engine(tts_mock=tts_mock)

        with pytest.raises(TTSError, match="TTS synthesis failed"):
            asyncio.run(
                engine.synthesize_speech("Hello")
            )


class TestEvaluateIntent:
    def test_allows_without_filter(self) -> None:
        engine = _create_engine()
        features = [_make_features()]
        decision = engine.evaluate_intent("anything", features)
        assert isinstance(decision, Decision)
        assert decision.allow is True

    def test_evaluates_with_filter(self) -> None:
        rules_path = str(
            Path(__file__).parent / "constitutional" / "sample_rules.yaml"
        )
        engine = _create_engine(constitutional_rules=rules_path)
        features = [_make_features()]
        decision = engine.evaluate_intent(
            "delete_account", features, emotion="sarcastic"
        )
        assert decision.allow is False


class TestTopLevelImports:
    def test_import_intent_engine(self) -> None:
        from intent_engine import IntentEngine as IE
        assert IE is IntentEngine

    def test_import_constitutional_filter(self) -> None:
        from intent_engine import ConstitutionalFilter
        assert ConstitutionalFilter is not None

    def test_import_errors(self) -> None:
        from intent_engine import IntentEngineError, LLMError, STTError, TTSError
        assert issubclass(STTError, IntentEngineError)
        assert issubclass(LLMError, IntentEngineError)
        assert issubclass(TTSError, IntentEngineError)

    def test_all_exports(self) -> None:
        import intent_engine
        expected = [
            "IntentEngine", "Result", "Response", "Audio", "Decision",
            "ConstitutionalFilter", "IntentEngineError", "STTError",
            "LLMError", "TTSError",
        ]
        for name in expected:
            assert name in intent_engine.__all__, f"Missing export: {name}"
