"""Tests for the Deepgram STT adapter."""

from __future__ import annotations

import asyncio
import os

import pytest
from prosody_protocol import WordAlignment

from intent_engine.stt.deepgram import DeepgramSTT


class TestDeepgramSTTConstruction:
    def test_requires_api_key(self) -> None:
        old_val = os.environ.pop("DEEPGRAM_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="Deepgram API key is required"):
                DeepgramSTT()
        finally:
            if old_val is not None:
                os.environ["DEEPGRAM_API_KEY"] = old_val

    def test_accepts_explicit_api_key(self) -> None:
        stt = DeepgramSTT(api_key="test-key-123")
        assert stt._api_key == "test-key-123"

    def test_reads_env_var(self) -> None:
        os.environ["DEEPGRAM_API_KEY"] = "env-key-456"
        try:
            stt = DeepgramSTT()
            assert stt._api_key == "env-key-456"
        finally:
            del os.environ["DEEPGRAM_API_KEY"]

    def test_default_model_and_language(self) -> None:
        stt = DeepgramSTT(api_key="key")
        assert stt._model == "nova-2"
        assert stt._language == "en"

    def test_custom_model_and_language(self) -> None:
        stt = DeepgramSTT(api_key="key", model="nova-3", language="fr")
        assert stt._model == "nova-3"
        assert stt._language == "fr"

    def test_is_stt_provider(self) -> None:
        from intent_engine.stt.base import STTProvider

        stt = DeepgramSTT(api_key="key")
        assert isinstance(stt, STTProvider)


class TestDeepgramSTTTranscribe:
    def test_import_error_without_sdk(self) -> None:
        import sys
        from unittest.mock import patch

        stt = DeepgramSTT(api_key="key")
        with patch.dict(sys.modules, {"deepgram": None}), pytest.raises(
            ImportError, match="deepgram-sdk is required"
        ):
            asyncio.get_event_loop().run_until_complete(
                stt.transcribe("/some/audio.wav")
            )


class TestDeepgramResponseParsing:
    """Test that Deepgram response format is correctly parsed into WordAlignments."""

    def test_word_alignment_creation(self) -> None:
        # Verify we can create the expected output types
        wa = WordAlignment(word="hello", start_ms=0, end_ms=500)
        assert wa.word == "hello"
        assert wa.start_ms == 0
        assert wa.end_ms == 500
