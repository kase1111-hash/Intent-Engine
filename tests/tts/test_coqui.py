"""Tests for the Coqui TTS adapter."""

from __future__ import annotations

import asyncio
import sys
import wave
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from intent_engine.tts.base import SynthesisResult, TTSProvider
from intent_engine.tts.coqui import CoquiTTS, _float_samples_to_wav


class TestCoquiTTSConstruction:
    def test_default_params(self) -> None:
        tts = CoquiTTS()
        assert tts._model_name == "tts_models/en/ljspeech/tacotron2-DDC"
        assert tts._device == "cpu"
        assert tts._speaker is None
        assert tts._language is None
        assert tts._tts is None

    def test_custom_params(self) -> None:
        tts = CoquiTTS(
            model_name="tts_models/en/vctk/vits",
            device="cuda",
            speaker="p225",
            language="en",
        )
        assert tts._model_name == "tts_models/en/vctk/vits"
        assert tts._device == "cuda"
        assert tts._speaker == "p225"
        assert tts._language == "en"

    def test_is_tts_provider(self) -> None:
        tts = CoquiTTS()
        assert isinstance(tts, TTSProvider)

    def test_accepts_kwargs(self) -> None:
        tts = CoquiTTS(extra_param="ignored")
        assert tts._model_name == "tts_models/en/ljspeech/tacotron2-DDC"

    def test_lazy_load_not_triggered_on_init(self) -> None:
        tts = CoquiTTS()
        assert tts._tts is None


class TestCoquiTTSLazyLoad:
    def test_import_error_without_tts(self) -> None:
        tts = CoquiTTS()
        with patch.dict(sys.modules, {"TTS": None, "TTS.api": None}), pytest.raises(
            ImportError, match="TTS .Coqui. is required"
        ):
            tts._load_model()


class TestCoquiTTSSynthesize:
    def test_synthesize_returns_synthesis_result(self) -> None:
        mock_tts_instance = MagicMock()
        # Return a list of float samples
        mock_tts_instance.tts.return_value = [0.0, 0.5, -0.5, 0.3, -0.1]

        tts = CoquiTTS()
        tts._tts = mock_tts_instance

        result = asyncio.run(
            tts.synthesize("Hello world", emotion="calm")
        )

        assert isinstance(result, SynthesisResult)
        assert result.format == "wav"
        assert result.sample_rate == 22050
        assert result.duration is not None
        assert result.duration > 0
        assert len(result.audio_data) > 0

    def test_synthesize_passes_speed_from_emotion(self) -> None:
        mock_tts_instance = MagicMock()
        mock_tts_instance.tts.return_value = [0.0, 0.1]

        tts = CoquiTTS()
        tts._tts = mock_tts_instance

        asyncio.run(
            tts.synthesize("Fast speech", emotion="angry")
        )

        call_kwargs = mock_tts_instance.tts.call_args.kwargs
        # angry has rate > 1.0
        assert call_kwargs["speed"] > 1.0

    def test_synthesize_with_speaker_and_language(self) -> None:
        mock_tts_instance = MagicMock()
        mock_tts_instance.tts.return_value = [0.0]

        tts = CoquiTTS(speaker="p225", language="en")
        tts._tts = mock_tts_instance

        asyncio.run(
            tts.synthesize("Test")
        )

        call_kwargs = mock_tts_instance.tts.call_args.kwargs
        assert call_kwargs["speaker"] == "p225"
        assert call_kwargs["language"] == "en"

    def test_output_is_valid_wav(self) -> None:
        mock_tts_instance = MagicMock()
        mock_tts_instance.tts.return_value = [0.0, 0.5, -0.5, 0.3]

        tts = CoquiTTS()
        tts._tts = mock_tts_instance

        result = asyncio.run(
            tts.synthesize("Hello")
        )

        # Verify it's a valid WAV file
        buf = BytesIO(result.audio_data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 22050
            assert wf.getnframes() == 4


class TestFloatSamplesToWav:
    def test_empty_samples(self) -> None:
        wav_bytes = _float_samples_to_wav([], 22050)
        buf = BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() == 0

    def test_sample_conversion(self) -> None:
        wav_bytes = _float_samples_to_wav([0.0, 1.0, -1.0], 22050)
        buf = BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() == 3
            assert wf.getsampwidth() == 2

    def test_clamping(self) -> None:
        # Values beyond [-1.0, 1.0] should be clamped
        wav_bytes = _float_samples_to_wav([2.0, -2.0], 22050)
        buf = BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() == 2
