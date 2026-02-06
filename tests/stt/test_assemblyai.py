"""Tests for the AssemblyAI STT adapter."""

from __future__ import annotations

import asyncio
import os

import pytest
from prosody_protocol import WordAlignment

from intent_engine.stt.assemblyai import AssemblyAISTT


class TestAssemblyAISTTConstruction:
    def test_requires_api_key(self) -> None:
        old_val = os.environ.pop("ASSEMBLYAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="AssemblyAI API key is required"):
                AssemblyAISTT()
        finally:
            if old_val is not None:
                os.environ["ASSEMBLYAI_API_KEY"] = old_val

    def test_accepts_explicit_api_key(self) -> None:
        stt = AssemblyAISTT(api_key="test-key-123")
        assert stt._api_key == "test-key-123"

    def test_reads_env_var(self) -> None:
        os.environ["ASSEMBLYAI_API_KEY"] = "env-key-456"
        try:
            stt = AssemblyAISTT()
            assert stt._api_key == "env-key-456"
        finally:
            del os.environ["ASSEMBLYAI_API_KEY"]

    def test_default_language(self) -> None:
        stt = AssemblyAISTT(api_key="key")
        assert stt._language_code == "en"

    def test_custom_language(self) -> None:
        stt = AssemblyAISTT(api_key="key", language_code="es")
        assert stt._language_code == "es"

    def test_is_stt_provider(self) -> None:
        from intent_engine.stt.base import STTProvider

        stt = AssemblyAISTT(api_key="key")
        assert isinstance(stt, STTProvider)


class TestAssemblyAISTTTranscribe:
    def test_import_error_without_sdk(self) -> None:
        import sys
        from unittest.mock import patch

        stt = AssemblyAISTT(api_key="key")
        with patch.dict(sys.modules, {"assemblyai": None}), pytest.raises(
            ImportError, match="assemblyai is required"
        ):
            asyncio.run(
                stt.transcribe("/some/audio.wav")
            )


class TestAssemblyAIResponseParsing:
    """Test that the expected output types are correct."""

    def test_word_alignment_creation(self) -> None:
        wa = WordAlignment(word="hello", start_ms=100, end_ms=500)
        assert wa.word == "hello"
        assert wa.start_ms == 100
        assert wa.end_ms == 500
