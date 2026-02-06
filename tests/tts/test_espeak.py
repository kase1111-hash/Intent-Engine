"""Tests for the eSpeak TTS adapter."""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from intent_engine.tts.base import SynthesisResult, TTSProvider
from intent_engine.tts.espeak import ESpeakTTS


class TestESpeakTTSConstruction:
    def test_default_params(self) -> None:
        tts = ESpeakTTS()
        assert tts._voice is None
        assert tts._rate_wpm == 175
        assert tts._volume == 1.0

    def test_custom_params(self) -> None:
        tts = ESpeakTTS(voice="english+f3", rate_wpm=200, volume=0.8)
        assert tts._voice == "english+f3"
        assert tts._rate_wpm == 200
        assert tts._volume == 0.8

    def test_is_tts_provider(self) -> None:
        tts = ESpeakTTS()
        assert isinstance(tts, TTSProvider)

    def test_accepts_kwargs(self) -> None:
        tts = ESpeakTTS(extra_param="ignored")
        assert tts._rate_wpm == 175


class TestESpeakTTSEngine:
    def test_import_error_without_pyttsx3(self) -> None:
        tts = ESpeakTTS()
        with patch.dict(sys.modules, {"pyttsx3": None}), pytest.raises(
            ImportError, match="pyttsx3 is required"
        ):
            tts._create_engine()

    def test_create_engine_sets_voice(self) -> None:
        mock_pyttsx3 = types.ModuleType("pyttsx3")
        mock_engine = MagicMock()
        mock_pyttsx3.init = MagicMock(return_value=mock_engine)  # type: ignore[attr-defined]
        sys.modules["pyttsx3"] = mock_pyttsx3

        try:
            tts = ESpeakTTS(voice="english+f3")
            engine = tts._create_engine()

            mock_engine.setProperty.assert_called_once_with("voice", "english+f3")
        finally:
            sys.modules.pop("pyttsx3", None)

    def test_create_engine_no_voice(self) -> None:
        mock_pyttsx3 = types.ModuleType("pyttsx3")
        mock_engine = MagicMock()
        mock_pyttsx3.init = MagicMock(return_value=mock_engine)  # type: ignore[attr-defined]
        sys.modules["pyttsx3"] = mock_pyttsx3

        try:
            tts = ESpeakTTS()
            engine = tts._create_engine()

            # setProperty should NOT be called for voice when voice is None
            mock_engine.setProperty.assert_not_called()
        finally:
            sys.modules.pop("pyttsx3", None)


class TestESpeakTTSSynthesize:
    def test_synthesize_adjusts_rate_for_emotion(self) -> None:
        mock_pyttsx3 = types.ModuleType("pyttsx3")
        mock_engine = MagicMock()
        mock_pyttsx3.init = MagicMock(return_value=mock_engine)  # type: ignore[attr-defined]
        sys.modules["pyttsx3"] = mock_pyttsx3

        try:
            tts = ESpeakTTS(rate_wpm=175)

            # Mock save_to_file to write some bytes
            def fake_save(text: str, path: str) -> None:
                from pathlib import Path
                Path(path).write_bytes(b"RIFF fake wav data")

            mock_engine.save_to_file.side_effect = fake_save

            result = asyncio.run(
                tts.synthesize("I'm angry!", emotion="angry")
            )

            assert isinstance(result, SynthesisResult)
            assert result.format == "wav"

            # Verify rate was adjusted (angry has rate > 1.0)
            rate_calls = [
                call for call in mock_engine.setProperty.call_args_list
                if call.args[0] == "rate"
            ]
            assert len(rate_calls) == 1
            adjusted_rate = rate_calls[0].args[1]
            assert adjusted_rate > 175  # angry = faster
        finally:
            sys.modules.pop("pyttsx3", None)

    def test_synthesize_adjusts_volume_for_emotion(self) -> None:
        mock_pyttsx3 = types.ModuleType("pyttsx3")
        mock_engine = MagicMock()
        mock_pyttsx3.init = MagicMock(return_value=mock_engine)  # type: ignore[attr-defined]
        sys.modules["pyttsx3"] = mock_pyttsx3

        try:
            tts = ESpeakTTS(volume=1.0)

            def fake_save(text: str, path: str) -> None:
                from pathlib import Path
                Path(path).write_bytes(b"RIFF fake wav data")

            mock_engine.save_to_file.side_effect = fake_save

            asyncio.run(
                tts.synthesize("I'm sad", emotion="sad")
            )

            # Verify volume was adjusted (sad has volume_db < 0)
            volume_calls = [
                call for call in mock_engine.setProperty.call_args_list
                if call.args[0] == "volume"
            ]
            assert len(volume_calls) == 1
            adjusted_volume = volume_calls[0].args[1]
            assert adjusted_volume < 1.0  # sad = quieter
        finally:
            sys.modules.pop("pyttsx3", None)

    def test_volume_clamped_to_valid_range(self) -> None:
        mock_pyttsx3 = types.ModuleType("pyttsx3")
        mock_engine = MagicMock()
        mock_pyttsx3.init = MagicMock(return_value=mock_engine)  # type: ignore[attr-defined]
        sys.modules["pyttsx3"] = mock_pyttsx3

        try:
            tts = ESpeakTTS(volume=1.0)

            def fake_save(text: str, path: str) -> None:
                from pathlib import Path
                Path(path).write_bytes(b"RIFF fake wav data")

            mock_engine.save_to_file.side_effect = fake_save

            # angry has +6dB which would push volume above 1.0
            asyncio.run(
                tts.synthesize("Loud", emotion="angry")
            )

            volume_calls = [
                call for call in mock_engine.setProperty.call_args_list
                if call.args[0] == "volume"
            ]
            assert len(volume_calls) == 1
            adjusted_volume = volume_calls[0].args[1]
            assert 0.0 <= adjusted_volume <= 1.0
        finally:
            sys.modules.pop("pyttsx3", None)

    def test_temp_file_cleaned_up(self) -> None:
        mock_pyttsx3 = types.ModuleType("pyttsx3")
        mock_engine = MagicMock()
        mock_pyttsx3.init = MagicMock(return_value=mock_engine)  # type: ignore[attr-defined]
        sys.modules["pyttsx3"] = mock_pyttsx3

        try:
            tts = ESpeakTTS()

            saved_path = None

            def fake_save(text: str, path: str) -> None:
                nonlocal saved_path
                saved_path = path
                from pathlib import Path
                Path(path).write_bytes(b"RIFF data")

            mock_engine.save_to_file.side_effect = fake_save

            asyncio.run(
                tts.synthesize("Test")
            )

            # Temp file should have been cleaned up
            from pathlib import Path
            assert saved_path is not None
            assert not Path(saved_path).exists()
        finally:
            sys.modules.pop("pyttsx3", None)
