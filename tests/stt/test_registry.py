"""Tests for the STT provider registry/factory."""

from __future__ import annotations

import os

import pytest

from intent_engine.stt import STT_PROVIDERS, create_stt_provider
from intent_engine.stt.base import STTProvider


class TestSTTProviderRegistry:
    def test_registry_has_all_providers(self) -> None:
        assert "whisper-prosody" in STT_PROVIDERS
        assert "deepgram" in STT_PROVIDERS
        assert "assemblyai" in STT_PROVIDERS

    def test_registry_has_three_entries(self) -> None:
        assert len(STT_PROVIDERS) == 3


class TestCreateSTTProvider:
    def test_create_whisper(self) -> None:
        stt = create_stt_provider("whisper-prosody")
        assert isinstance(stt, STTProvider)

    def test_create_whisper_with_params(self) -> None:
        stt = create_stt_provider("whisper-prosody", model_size="tiny", device="cpu")
        assert isinstance(stt, STTProvider)

    def test_create_deepgram(self) -> None:
        stt = create_stt_provider("deepgram", api_key="test-key")
        assert isinstance(stt, STTProvider)

    def test_create_assemblyai(self) -> None:
        stt = create_stt_provider("assemblyai", api_key="test-key")
        assert isinstance(stt, STTProvider)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown STT provider"):
            create_stt_provider("unknown-provider")

    def test_error_message_lists_available(self) -> None:
        with pytest.raises(ValueError, match="whisper-prosody"):
            create_stt_provider("bad")

    def test_deepgram_without_key_raises(self) -> None:
        old_val = os.environ.pop("DEEPGRAM_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="Deepgram API key"):
                create_stt_provider("deepgram")
        finally:
            if old_val is not None:
                os.environ["DEEPGRAM_API_KEY"] = old_val

    def test_assemblyai_without_key_raises(self) -> None:
        old_val = os.environ.pop("ASSEMBLYAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="AssemblyAI API key"):
                create_stt_provider("assemblyai")
        finally:
            if old_val is not None:
                os.environ["ASSEMBLYAI_API_KEY"] = old_val
