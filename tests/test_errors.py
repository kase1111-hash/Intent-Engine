"""Tests for the error hierarchy."""

from __future__ import annotations

import pytest

from intent_engine.errors import IntentEngineError, LLMError, STTError, TTSError


class TestErrorHierarchy:
    def test_stt_error_is_intent_engine_error(self) -> None:
        assert issubclass(STTError, IntentEngineError)

    def test_llm_error_is_intent_engine_error(self) -> None:
        assert issubclass(LLMError, IntentEngineError)

    def test_tts_error_is_intent_engine_error(self) -> None:
        assert issubclass(TTSError, IntentEngineError)

    def test_intent_engine_error_is_exception(self) -> None:
        assert issubclass(IntentEngineError, Exception)

    def test_catch_stt_as_base(self) -> None:
        with pytest.raises(IntentEngineError):
            raise STTError("transcription failed")

    def test_catch_llm_as_base(self) -> None:
        with pytest.raises(IntentEngineError):
            raise LLMError("interpretation failed")

    def test_catch_tts_as_base(self) -> None:
        with pytest.raises(IntentEngineError):
            raise TTSError("synthesis failed")

    def test_error_message(self) -> None:
        exc = STTError("test message")
        assert str(exc) == "test message"
