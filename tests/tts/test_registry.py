"""Tests for the TTS provider registry and factory."""

from __future__ import annotations

import os

import pytest

from intent_engine.tts import TTS_PROVIDERS, TTSProvider, create_tts_provider


class TestTTSProviderRegistry:
    def test_registry_has_expected_providers(self) -> None:
        assert "elevenlabs" in TTS_PROVIDERS
        assert "coqui" in TTS_PROVIDERS
        assert "espeak" in TTS_PROVIDERS

    def test_registry_has_three_providers(self) -> None:
        assert len(TTS_PROVIDERS) == 3

    def test_registry_values_are_module_paths(self) -> None:
        for name, path in TTS_PROVIDERS.items():
            assert path.startswith("intent_engine.tts."), (
                f"Provider {name} path should start with intent_engine.tts."
            )


class TestCreateTTSProvider:
    def test_create_elevenlabs_provider(self) -> None:
        tts = create_tts_provider("elevenlabs", api_key="test-key")
        assert isinstance(tts, TTSProvider)
        from intent_engine.tts.elevenlabs import ElevenLabsTTS

        assert isinstance(tts, ElevenLabsTTS)

    def test_create_coqui_provider(self) -> None:
        tts = create_tts_provider("coqui")
        assert isinstance(tts, TTSProvider)
        from intent_engine.tts.coqui import CoquiTTS

        assert isinstance(tts, CoquiTTS)

    def test_create_espeak_provider(self) -> None:
        tts = create_tts_provider("espeak")
        assert isinstance(tts, TTSProvider)
        from intent_engine.tts.espeak import ESpeakTTS

        assert isinstance(tts, ESpeakTTS)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            create_tts_provider("nonexistent")

    def test_error_message_lists_available_providers(self) -> None:
        with pytest.raises(ValueError, match="elevenlabs.*coqui.*espeak"):
            create_tts_provider("bad-name")

    def test_kwargs_passed_to_constructor(self) -> None:
        tts = create_tts_provider(
            "elevenlabs",
            api_key="my-key",
            voice_id="custom-voice",
            model_id="eleven_multilingual_v2",
        )
        from intent_engine.tts.elevenlabs import ElevenLabsTTS

        assert isinstance(tts, ElevenLabsTTS)
        assert tts._api_key == "my-key"
        assert tts._voice_id == "custom-voice"
        assert tts._model_id == "eleven_multilingual_v2"

    def test_elevenlabs_without_key_raises(self) -> None:
        old_val = os.environ.pop("ELEVENLABS_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="ElevenLabs API key"):
                create_tts_provider("elevenlabs")
        finally:
            if old_val is not None:
                os.environ["ELEVENLABS_API_KEY"] = old_val

    def test_coqui_no_key_needed(self) -> None:
        tts = create_tts_provider("coqui")
        assert tts is not None

    def test_espeak_no_key_needed(self) -> None:
        tts = create_tts_provider("espeak")
        assert tts is not None
