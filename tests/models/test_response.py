"""Tests for the Response dataclass."""

from __future__ import annotations

import dataclasses

from intent_engine.models.response import Response


class TestResponseConstruction:
    def test_basic_construction(self) -> None:
        resp = Response(text="I understand your frustration.", emotion="empathetic")
        assert resp.text == "I understand your frustration."
        assert resp.emotion == "empathetic"

    def test_with_core_emotions(self) -> None:
        for emotion in ["neutral", "sarcastic", "frustrated", "joyful", "calm", "angry", "sad"]:
            resp = Response(text="test", emotion=emotion)
            assert resp.emotion == emotion


class TestResponseIntent:
    def test_intent_defaults_to_none(self) -> None:
        resp = Response(text="hello", emotion="calm")
        assert resp.intent is None

    def test_intent_is_the_trailing_field(self) -> None:
        assert [f.name for f in dataclasses.fields(Response)] == ["text", "emotion", "intent"]

    def test_positional_construction_is_unchanged(self) -> None:
        resp = Response("hello", "calm")
        assert (resp.text, resp.emotion, resp.intent) == ("hello", "calm", None)

    def test_with_intent(self) -> None:
        resp = Response(text="Are you sure?", emotion="calm", intent="delete_files")
        assert resp.intent == "delete_files"

    def test_intent_takes_part_in_equality(self) -> None:
        a = Response(text="hello", emotion="calm", intent="greet")
        b = Response(text="hello", emotion="calm", intent="cancel")
        assert a != b


class TestResponseImmutability:
    def test_frozen(self) -> None:
        resp = Response(text="test", emotion="neutral")
        assert dataclasses.is_dataclass(resp)
        try:
            resp.text = "changed"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except dataclasses.FrozenInstanceError:
            pass

    def test_equality(self) -> None:
        a = Response(text="hello", emotion="calm")
        b = Response(text="hello", emotion="calm")
        assert a == b

    def test_inequality(self) -> None:
        a = Response(text="hello", emotion="calm")
        b = Response(text="hello", emotion="angry")
        assert a != b
