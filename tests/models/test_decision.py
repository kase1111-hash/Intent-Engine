"""Tests for the Decision dataclass."""

from __future__ import annotations

import dataclasses

from intent_engine.models.decision import Decision


class TestDecisionConstruction:
    def test_allow(self) -> None:
        d = Decision(allow=True)
        assert d.allow is True
        assert d.requires_verification is False
        assert d.verification_method is None
        assert d.denial_reason is None

    def test_deny(self) -> None:
        d = Decision(allow=False, denial_reason="Sarcastic tone detected")
        assert d.allow is False
        assert d.denial_reason == "Sarcastic tone detected"

    def test_requires_verification(self) -> None:
        d = Decision(
            allow=False,
            requires_verification=True,
            verification_method="explicit_confirmation",
            denial_reason="Prosody check failed: emotion not in [calm, confident]",
        )
        assert d.allow is False
        assert d.requires_verification is True
        assert d.verification_method == "explicit_confirmation"
        assert "Prosody check failed" in (d.denial_reason or "")

    def test_two_factor(self) -> None:
        d = Decision(
            allow=False,
            requires_verification=True,
            verification_method="two_factor",
        )
        assert d.verification_method == "two_factor"


class TestDecisionImmutability:
    def test_frozen(self) -> None:
        d = Decision(allow=True)
        assert dataclasses.is_dataclass(d)
        try:
            d.allow = False  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except dataclasses.FrozenInstanceError:
            pass


class TestDecisionEquality:
    def test_equal_decisions(self) -> None:
        a = Decision(allow=True)
        b = Decision(allow=True)
        assert a == b

    def test_unequal_decisions(self) -> None:
        a = Decision(allow=True)
        b = Decision(allow=False, denial_reason="denied")
        assert a != b
