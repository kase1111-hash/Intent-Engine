"""Tests for the ElevenLabs TTS adapter."""

from __future__ import annotations

import asyncio
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from intent_engine.tts.base import SynthesisResult, TTSProvider
from intent_engine.tts.elevenlabs import ELEVENLABS_EMOTION_SETTINGS, ElevenLabsTTS


class TestElevenLabsTTSConstruction:
    def test_requires_api_key(self) -> None:
        old_val = os.environ.pop("ELEVENLABS_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="ElevenLabs API key is required"):
                ElevenLabsTTS()
        finally:
            if old_val is not None:
                os.environ["ELEVENLABS_API_KEY"] = old_val

    def test_accepts_explicit_api_key(self) -> None:
        tts = ElevenLabsTTS(api_key="test-key-123")
        assert tts._api_key == "test-key-123"

    def test_reads_env_var(self) -> None:
        os.environ["ELEVENLABS_API_KEY"] = "env-key-456"
        try:
            tts = ElevenLabsTTS()
            assert tts._api_key == "env-key-456"
        finally:
            del os.environ["ELEVENLABS_API_KEY"]

    def test_default_params(self) -> None:
        tts = ElevenLabsTTS(api_key="key")
        assert tts._voice_id == "21m00Tcm4TlvDq8ikWAM"
        assert tts._model_id == "eleven_monolingual_v1"
        assert tts._output_format == "mp3_44100_128"

    def test_custom_params(self) -> None:
        tts = ElevenLabsTTS(
            api_key="key",
            voice_id="custom-voice-id",
            model_id="eleven_multilingual_v2",
            output_format="pcm_22050",
        )
        assert tts._voice_id == "custom-voice-id"
        assert tts._model_id == "eleven_multilingual_v2"
        assert tts._output_format == "pcm_22050"

    def test_is_tts_provider(self) -> None:
        tts = ElevenLabsTTS(api_key="key")
        assert isinstance(tts, TTSProvider)

    def test_accepts_kwargs(self) -> None:
        tts = ElevenLabsTTS(api_key="key", extra_param="ignored")
        assert tts._api_key == "key"


class TestElevenLabsEmotionSettings:
    CORE_EMOTIONS = [
        "neutral", "sincere", "sarcastic", "frustrated", "joyful",
        "uncertain", "angry", "sad", "fearful", "surprised",
        "disgusted", "calm", "empathetic",
    ]

    def test_all_core_emotions_mapped(self) -> None:
        for emotion in self.CORE_EMOTIONS:
            assert emotion in ELEVENLABS_EMOTION_SETTINGS, (
                f"Missing ElevenLabs mapping for: {emotion}"
            )

    def test_settings_have_required_keys(self) -> None:
        for emotion, settings in ELEVENLABS_EMOTION_SETTINGS.items():
            assert "stability" in settings, f"{emotion} missing stability"
            assert "similarity_boost" in settings, f"{emotion} missing similarity_boost"
            assert "style" in settings, f"{emotion} missing style"

    def test_values_in_range(self) -> None:
        for emotion, settings in ELEVENLABS_EMOTION_SETTINGS.items():
            assert 0.0 <= settings["stability"] <= 1.0, (
                f"{emotion} stability out of range"
            )
            assert 0.0 <= settings["similarity_boost"] <= 1.0, (
                f"{emotion} similarity_boost out of range"
            )
            assert 0.0 <= settings["style"] <= 1.0, (
                f"{emotion} style out of range"
            )

    def test_angry_has_high_style(self) -> None:
        assert ELEVENLABS_EMOTION_SETTINGS["angry"]["style"] > 0.7

    def test_calm_has_high_stability(self) -> None:
        assert ELEVENLABS_EMOTION_SETTINGS["calm"]["stability"] > 0.6


class TestElevenLabsTTSSynthesize:
    def test_import_error_without_sdk(self) -> None:
        tts = ElevenLabsTTS(api_key="key")
        with patch.dict(sys.modules, {"elevenlabs": None}), pytest.raises(
            ImportError, match="elevenlabs is required"
        ):
            asyncio.get_event_loop().run_until_complete(
                tts.synthesize("Hello")
            )

    def test_synthesize_returns_synthesis_result(self) -> None:
        mock_elevenlabs = types.ModuleType("elevenlabs")

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"audio", b"bytes"])

        mock_elevenlabs.ElevenLabs = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
        sys.modules["elevenlabs"] = mock_elevenlabs

        try:
            tts = ElevenLabsTTS(api_key="test-key")
            result = asyncio.get_event_loop().run_until_complete(
                tts.synthesize("Hello world", emotion="joyful")
            )

            assert isinstance(result, SynthesisResult)
            assert result.audio_data == b"audiobytes"
            assert result.format == "mp3"
            assert result.sample_rate == 44100
        finally:
            sys.modules.pop("elevenlabs", None)

    def test_synthesize_passes_emotion_settings(self) -> None:
        mock_elevenlabs = types.ModuleType("elevenlabs")

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"data"])

        mock_elevenlabs.ElevenLabs = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
        sys.modules["elevenlabs"] = mock_elevenlabs

        try:
            tts = ElevenLabsTTS(api_key="test-key")
            asyncio.get_event_loop().run_until_complete(
                tts.synthesize("I'm frustrated", emotion="frustrated")
            )

            call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
            voice_settings = call_kwargs["voice_settings"]
            expected = ELEVENLABS_EMOTION_SETTINGS["frustrated"]
            assert voice_settings["stability"] == expected["stability"]
            assert voice_settings["similarity_boost"] == expected["similarity_boost"]
            assert voice_settings["style"] == expected["style"]
        finally:
            sys.modules.pop("elevenlabs", None)

    def test_unknown_emotion_uses_neutral_settings(self) -> None:
        mock_elevenlabs = types.ModuleType("elevenlabs")

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"data"])

        mock_elevenlabs.ElevenLabs = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
        sys.modules["elevenlabs"] = mock_elevenlabs

        try:
            tts = ElevenLabsTTS(api_key="test-key")
            asyncio.get_event_loop().run_until_complete(
                tts.synthesize("Test", emotion="unknown_emotion")
            )

            call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
            voice_settings = call_kwargs["voice_settings"]
            expected = ELEVENLABS_EMOTION_SETTINGS["neutral"]
            assert voice_settings["stability"] == expected["stability"]
        finally:
            sys.modules.pop("elevenlabs", None)
