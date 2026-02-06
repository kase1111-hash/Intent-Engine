"""Tests for STT base classes."""

from __future__ import annotations

import dataclasses

from prosody_protocol import WordAlignment

from intent_engine.stt.base import STTProvider, TranscriptionResult


class TestTranscriptionResult:
    def test_construction(self) -> None:
        alignments = [
            WordAlignment(word="Hello", start_ms=0, end_ms=500),
            WordAlignment(word="world", start_ms=500, end_ms=1000),
        ]
        result = TranscriptionResult(text="Hello world", alignments=alignments)
        assert result.text == "Hello world"
        assert len(result.alignments) == 2
        assert result.language is None

    def test_with_language(self) -> None:
        result = TranscriptionResult(text="Bonjour", alignments=[], language="fr")
        assert result.language == "fr"

    def test_frozen(self) -> None:
        result = TranscriptionResult(text="test", alignments=[])
        try:
            result.text = "changed"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except dataclasses.FrozenInstanceError:
            pass

    def test_word_alignment_types(self) -> None:
        wa = WordAlignment(word="test", start_ms=100, end_ms=200)
        assert isinstance(wa, WordAlignment)
        assert wa.word == "test"
        assert wa.start_ms == 100
        assert wa.end_ms == 200

    def test_empty_alignments(self) -> None:
        result = TranscriptionResult(text="", alignments=[])
        assert result.text == ""
        assert result.alignments == []


class TestSTTProviderInterface:
    def test_is_abstract(self) -> None:
        try:
            STTProvider()  # type: ignore[abstract]
            raise AssertionError("Should have raised TypeError")
        except TypeError:
            pass

    def test_subclass_must_implement_transcribe(self) -> None:
        class IncompleteSTT(STTProvider):
            pass

        try:
            IncompleteSTT()  # type: ignore[abstract]
            raise AssertionError("Should have raised TypeError")
        except TypeError:
            pass

    def test_valid_subclass(self) -> None:
        class MockSTT(STTProvider):
            async def transcribe(self, audio_path: str) -> TranscriptionResult:
                return TranscriptionResult(text="mock", alignments=[])

        stt = MockSTT()
        assert isinstance(stt, STTProvider)
