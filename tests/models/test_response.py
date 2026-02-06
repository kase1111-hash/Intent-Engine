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
