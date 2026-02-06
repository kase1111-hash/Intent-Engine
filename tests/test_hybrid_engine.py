"""Tests for HybridEngine (cloud STT/TTS + local LLM deployment)."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prosody_protocol import (
    IMLDocument,
    Segment,
    SpanFeatures,
    Utterance,
    ValidationResult,
)

from intent_engine.engine import IntentEngine
from intent_engine.hybrid_engine import HybridEngine, _HYBRID_DEFAULTS
from intent_engine.llm.base import InterpretationResult
from intent_engine.models.response import Response
from intent_engine.models.result import Result
from intent_engine.stt.base import TranscriptionResult
from intent_engine.tts.base import SynthesisResult


def _create_hybrid(
    stt_mock: MagicMock | None = None,
    llm_mock: MagicMock | None = None,
    tts_mock: MagicMock | None = None,
    **kwargs,
) -> HybridEngine:
    """Create a HybridEngine with mocked providers."""
    with patch("intent_engine.engine.create_stt_provider") as mock_stt_factory, \
         patch("intent_engine.engine.create_llm_provider") as mock_llm_factory, \
         patch("intent_engine.engine.create_tts_provider") as mock_tts_factory:
        mock_stt_factory.return_value = stt_mock or MagicMock()
        mock_llm_factory.return_value = llm_mock or MagicMock()
        mock_tts_factory.return_value = tts_mock or MagicMock()
        engine = HybridEngine(**kwargs)
    return engine


class TestHybridEngineConstruction:
    def test_creates_successfully(self) -> None:
        engine = _create_hybrid()
        assert engine is not None

    def test_inherits_intent_engine(self) -> None:
        engine = _create_hybrid()
        assert isinstance(engine, IntentEngine)

    def test_default_providers(self) -> None:
        with patch("intent_engine.engine.create_stt_provider") as stt_f, \
             patch("intent_engine.engine.create_llm_provider") as llm_f, \
             patch("intent_engine.engine.create_tts_provider") as tts_f:
            stt_f.return_value = MagicMock()
            llm_f.return_value = MagicMock()
            tts_f.return_value = MagicMock()
            HybridEngine()
            stt_f.assert_called_once()
            assert stt_f.call_args[0][0] == "deepgram"
            llm_f.assert_called_once()
            assert llm_f.call_args[0][0] == "local"
            tts_f.assert_called_once()
            assert tts_f.call_args[0][0] == "coqui"

    def test_custom_providers(self) -> None:
        with patch("intent_engine.engine.create_stt_provider") as stt_f, \
             patch("intent_engine.engine.create_llm_provider") as llm_f, \
             patch("intent_engine.engine.create_tts_provider") as tts_f:
            stt_f.return_value = MagicMock()
            llm_f.return_value = MagicMock()
            tts_f.return_value = MagicMock()
            HybridEngine(
                stt_provider="assemblyai",
                tts_provider="elevenlabs",
                stt_kwargs={"api_key": "test"},
                tts_kwargs={"api_key": "test"},
            )
            assert stt_f.call_args[0][0] == "assemblyai"
            assert tts_f.call_args[0][0] == "elevenlabs"

    def test_llm_model_passed_to_kwargs(self) -> None:
        with patch("intent_engine.engine.create_stt_provider") as stt_f, \
             patch("intent_engine.engine.create_llm_provider") as llm_f, \
             patch("intent_engine.engine.create_tts_provider") as tts_f:
            stt_f.return_value = MagicMock()
            llm_f.return_value = MagicMock()
            tts_f.return_value = MagicMock()
            HybridEngine(llm_model="models/llama-3.1-70b-prosody-ft.gguf")
            call_kwargs = llm_f.call_args[1]
            assert call_kwargs.get("model_path") == "models/llama-3.1-70b-prosody-ft.gguf"


class TestHybridEngineProperties:
    def test_llm_model_property(self) -> None:
        engine = _create_hybrid(llm_model="my-model.gguf")
        assert engine.llm_model == "my-model.gguf"

    def test_llm_model_none(self) -> None:
        engine = _create_hybrid()
        assert engine.llm_model is None

    def test_is_llm_local(self) -> None:
        engine = _create_hybrid()
        assert engine.is_llm_local is True

    def test_deployment_mode(self) -> None:
        engine = _create_hybrid()
        assert engine.deployment_mode == "hybrid"


class TestHybridEnginePipeline:
    def test_inherits_process_voice_input(self) -> None:
        """HybridEngine should have process_voice_input from IntentEngine."""
        stt_mock = MagicMock()
        stt_mock.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="Test", alignments=[], language="en")
        )

        engine = _create_hybrid(stt_mock=stt_mock)
        engine._analyzer.analyze = MagicMock(return_value=[])
        engine._analyzer.detect_pauses = MagicMock(return_value=[])

        iml_doc = IMLDocument(
            utterances=(Utterance(children=(Segment(),)),), version="1.0"
        )
        engine._assembler.assemble = MagicMock(return_value=iml_doc)
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
            result = asyncio.get_event_loop().run_until_complete(
                engine.process_voice_input(audio_path)
            )
            assert isinstance(result, Result)
            assert result.text == "Test"
        finally:
            Path(audio_path).unlink()

    def test_inherits_generate_response(self) -> None:
        llm_mock = MagicMock()
        llm_mock.interpret = AsyncMock(
            return_value=InterpretationResult(
                intent="greet", response_text="Hi!", suggested_emotion="joyful"
            )
        )

        engine = _create_hybrid(llm_mock=llm_mock)

        result = asyncio.get_event_loop().run_until_complete(
            engine.generate_response("<utterance>Hello</utterance>")
        )
        assert isinstance(result, Response)
        assert result.text == "Hi!"

    def test_has_prosody_protocol_components(self) -> None:
        engine = _create_hybrid()
        assert engine._analyzer is not None
        assert engine._assembler is not None
        assert engine._parser is not None
        assert engine._validator is not None
        assert engine._emotion_classifier is not None
