"""Tests for LocalEngine (fully local deployment)."""

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
from intent_engine.local_engine import HARDWARE_TIERS, LocalEngine, _LOCAL_DEFAULTS
from intent_engine.llm.base import InterpretationResult
from intent_engine.models.audio import Audio
from intent_engine.models.response import Response
from intent_engine.models.result import Result
from intent_engine.stt.base import TranscriptionResult
from intent_engine.tts.base import SynthesisResult


def _create_local(
    stt_mock: MagicMock | None = None,
    llm_mock: MagicMock | None = None,
    tts_mock: MagicMock | None = None,
    **kwargs,
) -> LocalEngine:
    """Create a LocalEngine with mocked providers."""
    kwargs.setdefault("validate_models", False)
    with patch("intent_engine.engine.create_stt_provider") as mock_stt_factory, \
         patch("intent_engine.engine.create_llm_provider") as mock_llm_factory, \
         patch("intent_engine.engine.create_tts_provider") as mock_tts_factory:
        mock_stt_factory.return_value = stt_mock or MagicMock()
        mock_llm_factory.return_value = llm_mock or MagicMock()
        mock_tts_factory.return_value = tts_mock or MagicMock()
        engine = LocalEngine(**kwargs)
    return engine


class TestLocalEngineConstruction:
    def test_creates_successfully(self) -> None:
        engine = _create_local()
        assert engine is not None

    def test_inherits_intent_engine(self) -> None:
        engine = _create_local()
        assert isinstance(engine, IntentEngine)

    def test_default_providers(self) -> None:
        with patch("intent_engine.engine.create_stt_provider") as stt_f, \
             patch("intent_engine.engine.create_llm_provider") as llm_f, \
             patch("intent_engine.engine.create_tts_provider") as tts_f:
            stt_f.return_value = MagicMock()
            llm_f.return_value = MagicMock()
            tts_f.return_value = MagicMock()
            LocalEngine(validate_models=False)
            assert stt_f.call_args[0][0] == "whisper-prosody"
            assert llm_f.call_args[0][0] == "local"
            assert tts_f.call_args[0][0] == "espeak"

    def test_custom_providers(self) -> None:
        with patch("intent_engine.engine.create_stt_provider") as stt_f, \
             patch("intent_engine.engine.create_llm_provider") as llm_f, \
             patch("intent_engine.engine.create_tts_provider") as tts_f:
            stt_f.return_value = MagicMock()
            llm_f.return_value = MagicMock()
            tts_f.return_value = MagicMock()
            LocalEngine(tts_provider="coqui", validate_models=False)
            assert tts_f.call_args[0][0] == "coqui"


class TestModelValidation:
    def test_validates_gguf_path_exists(self) -> None:
        """GGUF file path that does not exist should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="llm_model path does not exist"):
            _create_local(
                llm_model="/nonexistent/path/model.gguf",
                validate_models=True,
            )

    def test_validates_path_with_slash(self) -> None:
        """Paths with slashes should be validated."""
        with pytest.raises(FileNotFoundError, match="stt_model path does not exist"):
            _create_local(
                stt_model="/nonexistent/whisper/model",
                validate_models=True,
            )

    def test_skips_named_models(self) -> None:
        """Named models like 'whisper-large-v3' should not be validated."""
        # Should NOT raise even though 'whisper-large-v3' is not a real file
        engine = _create_local(
            stt_model="whisper-large-v3",
            llm_model="llama-3.1-70b",
            validate_models=True,
        )
        assert engine.stt_model == "whisper-large-v3"

    def test_skips_validation_when_disabled(self) -> None:
        """Should not raise when validate_models=False."""
        engine = _create_local(
            llm_model="/nonexistent/model.gguf",
            validate_models=False,
        )
        assert engine.llm_model == "/nonexistent/model.gguf"

    def test_validates_real_file_path(self) -> None:
        """A real file path should pass validation."""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"fake model data")
            model_path = f.name

        try:
            engine = _create_local(llm_model=model_path, validate_models=True)
            assert engine.llm_model == model_path
        finally:
            Path(model_path).unlink()

    def test_model_extensions_recognized(self) -> None:
        """Files ending in model extensions should be validated."""
        for ext in (".gguf", ".bin", ".pt", ".pth", ".onnx", ".safetensors"):
            with pytest.raises(FileNotFoundError):
                _create_local(
                    llm_model=f"model{ext}",
                    validate_models=True,
                )


class TestLocalEngineProperties:
    def test_stt_model(self) -> None:
        engine = _create_local(stt_model="whisper-large-v3")
        assert engine.stt_model == "whisper-large-v3"

    def test_llm_model(self) -> None:
        engine = _create_local(llm_model="llama-3.1-70b")
        assert engine.llm_model == "llama-3.1-70b"

    def test_tts_model(self) -> None:
        engine = _create_local(tts_model="coqui-tts-v1")
        assert engine.tts_model == "coqui-tts-v1"

    def test_prosody_model(self) -> None:
        engine = _create_local(prosody_model="prosody-analyzer-v2")
        assert engine.prosody_model == "prosody-analyzer-v2"

    def test_is_fully_local(self) -> None:
        engine = _create_local()
        assert engine.is_fully_local is True

    def test_deployment_mode(self) -> None:
        engine = _create_local()
        assert engine.deployment_mode == "local"

    def test_models_default_to_none(self) -> None:
        engine = _create_local()
        assert engine.stt_model is None
        assert engine.llm_model is None
        assert engine.tts_model is None
        assert engine.prosody_model is None


class TestLocalModelKwargsPassthrough:
    def test_stt_model_in_kwargs(self) -> None:
        with patch("intent_engine.engine.create_stt_provider") as stt_f, \
             patch("intent_engine.engine.create_llm_provider") as llm_f, \
             patch("intent_engine.engine.create_tts_provider") as tts_f:
            stt_f.return_value = MagicMock()
            llm_f.return_value = MagicMock()
            tts_f.return_value = MagicMock()
            LocalEngine(stt_model="whisper-large-v3", validate_models=False)
            call_kwargs = stt_f.call_args[1]
            assert call_kwargs.get("model") == "whisper-large-v3"

    def test_llm_model_in_kwargs(self) -> None:
        with patch("intent_engine.engine.create_stt_provider") as stt_f, \
             patch("intent_engine.engine.create_llm_provider") as llm_f, \
             patch("intent_engine.engine.create_tts_provider") as tts_f:
            stt_f.return_value = MagicMock()
            llm_f.return_value = MagicMock()
            tts_f.return_value = MagicMock()
            LocalEngine(llm_model="llama-3.1-70b", validate_models=False)
            call_kwargs = llm_f.call_args[1]
            assert call_kwargs.get("model_path") == "llama-3.1-70b"

    def test_tts_model_in_kwargs(self) -> None:
        with patch("intent_engine.engine.create_stt_provider") as stt_f, \
             patch("intent_engine.engine.create_llm_provider") as llm_f, \
             patch("intent_engine.engine.create_tts_provider") as tts_f:
            stt_f.return_value = MagicMock()
            llm_f.return_value = MagicMock()
            tts_f.return_value = MagicMock()
            LocalEngine(tts_model="coqui-tts-v1", validate_models=False)
            call_kwargs = tts_f.call_args[1]
            assert call_kwargs.get("model") == "coqui-tts-v1"


class TestLocalEnginePipeline:
    def test_inherits_process_voice_input(self) -> None:
        stt_mock = MagicMock()
        stt_mock.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="Hello", alignments=[], language="en")
        )

        engine = _create_local(stt_mock=stt_mock)
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
            assert result.text == "Hello"
        finally:
            Path(audio_path).unlink()

    def test_inherits_generate_response(self) -> None:
        llm_mock = MagicMock()
        llm_mock.interpret = AsyncMock(
            return_value=InterpretationResult(
                intent="greet", response_text="Hi!", suggested_emotion="neutral"
            )
        )

        engine = _create_local(llm_mock=llm_mock)

        result = asyncio.get_event_loop().run_until_complete(
            engine.generate_response("<utterance>Hello</utterance>")
        )
        assert isinstance(result, Response)
        assert result.text == "Hi!"

    def test_inherits_synthesize_speech(self) -> None:
        tts_mock = MagicMock()
        tts_mock.synthesize = AsyncMock(
            return_value=SynthesisResult(
                audio_data=b"fake audio", format="wav", sample_rate=22050, duration=1.0
            )
        )

        engine = _create_local(tts_mock=tts_mock)

        result = asyncio.get_event_loop().run_until_complete(
            engine.synthesize_speech("Hello", emotion="joyful")
        )
        assert isinstance(result, Audio)
        assert result.data == b"fake audio"


class TestHardwareTiers:
    def test_tiers_defined(self) -> None:
        assert "minimum" in HARDWARE_TIERS
        assert "recommended" in HARDWARE_TIERS
        assert "optimal" in HARDWARE_TIERS

    def test_minimum_tier(self) -> None:
        tier = HARDWARE_TIERS["minimum"]
        assert tier["ram_gb"] == 16
        assert tier["gpu"] == "CPU-only"

    def test_optimal_tier(self) -> None:
        tier = HARDWARE_TIERS["optimal"]
        assert tier["ram_gb"] == 128


class TestTopLevelImports:
    def test_import_cloud_engine(self) -> None:
        from intent_engine import CloudEngine
        assert CloudEngine is not None

    def test_import_hybrid_engine(self) -> None:
        from intent_engine import HybridEngine
        assert HybridEngine is not None

    def test_import_local_engine(self) -> None:
        from intent_engine import LocalEngine
        assert LocalEngine is not None

    def test_all_exports(self) -> None:
        import intent_engine
        for name in ["CloudEngine", "HybridEngine", "LocalEngine"]:
            assert name in intent_engine.__all__, f"Missing export: {name}"
