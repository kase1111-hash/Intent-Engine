"""Tests for the Result dataclass."""

from __future__ import annotations

import dataclasses

from prosody_protocol import IMLDocument, SpanFeatures, Utterance

from intent_engine.models.result import Result


def _make_document() -> IMLDocument:
    return IMLDocument(
        utterances=(
            Utterance(children=("Hello world",), emotion="neutral", confidence=0.9),
        ),
    )


def _make_features() -> list[SpanFeatures]:
    return [
        SpanFeatures(start_ms=0, end_ms=500, text="Hello", f0_mean=120.0),
        SpanFeatures(start_ms=500, end_ms=1000, text="world", f0_mean=115.0),
    ]


class TestResultConstruction:
    def test_basic_construction(self) -> None:
        doc = _make_document()
        features = _make_features()
        result = Result(
            text="Hello world",
            emotion="neutral",
            confidence=0.9,
            iml='<utterance emotion="neutral" confidence="0.9">Hello world</utterance>',
            iml_document=doc,
            suggested_tone="neutral",
            prosody_features=features,
        )
        assert result.text == "Hello world"
        assert result.emotion == "neutral"
        assert result.confidence == 0.9
        assert result.suggested_tone == "neutral"
        assert result.intent is None

    def test_with_intent(self) -> None:
        doc = _make_document()
        result = Result(
            text="Delete all files",
            emotion="calm",
            confidence=0.85,
            iml="<utterance>Delete all files</utterance>",
            iml_document=doc,
            suggested_tone="calm",
            prosody_features=[],
            intent="delete_files",
        )
        assert result.intent == "delete_files"


class TestResultImmutability:
    def test_frozen(self) -> None:
        doc = _make_document()
        result = Result(
            text="test",
            emotion="neutral",
            confidence=0.5,
            iml="<utterance>test</utterance>",
            iml_document=doc,
            suggested_tone="neutral",
            prosody_features=[],
        )
        assert dataclasses.is_dataclass(result)
        try:
            result.text = "changed"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except dataclasses.FrozenInstanceError:
            pass


class TestResultProsodyProtocolTypes:
    def test_iml_document_type(self) -> None:
        doc = _make_document()
        result = Result(
            text="test",
            emotion="neutral",
            confidence=0.5,
            iml="<utterance>test</utterance>",
            iml_document=doc,
            suggested_tone="neutral",
            prosody_features=[],
        )
        assert isinstance(result.iml_document, IMLDocument)
        assert len(result.iml_document.utterances) == 1
        assert result.iml_document.utterances[0].emotion == "neutral"

    def test_span_features_type(self) -> None:
        doc = _make_document()
        features = _make_features()
        result = Result(
            text="Hello world",
            emotion="neutral",
            confidence=0.9,
            iml="<utterance>Hello world</utterance>",
            iml_document=doc,
            suggested_tone="neutral",
            prosody_features=features,
        )
        assert len(result.prosody_features) == 2
        assert isinstance(result.prosody_features[0], SpanFeatures)
        assert result.prosody_features[0].text == "Hello"
        assert result.prosody_features[0].f0_mean == 120.0
        assert result.prosody_features[1].start_ms == 500
