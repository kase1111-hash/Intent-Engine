"""Emotion abstention: low confidence means "unknown", not "neutral" (audit #36).

Prosody Protocol 0.1.0a3 abstains from labelling with ``("neutral", 0.0)``
when it cannot judge.  Passed on as evidence, that abstention satisfied
rules that require "neutral" and never triggered forbidden-emotion rules.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from prosody_protocol import SpanFeatures

from intent_engine.constitutional import ConstitutionalFilter
from intent_engine.constitutional.evaluator import resolve_emotion
from intent_engine.constitutional.rules import ConstitutionalRule, ProsodyCondition, Verification

SAMPLE_RULES_PATH = Path(__file__).parent / "sample_rules.yaml"
FEATURES = [SpanFeatures(0, 500, "w", f0_mean=120.0)]


@pytest.fixture()
def cf() -> ConstitutionalFilter:
    return ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)


class TestAbstention:
    def test_abstained_neutral_is_not_positive_evidence(self, cf: ConstitutionalFilter) -> None:
        """financial_transaction accepts sincere/calm/neutral, but not "no reading"."""
        decision = cf.evaluate(
            "send_money", FEATURES, emotion="neutral", emotion_confidence=0.0
        )
        assert decision.allow is False
        assert decision.requires_verification is True
        assert decision.verification_method == "two_factor"

    def test_confident_neutral_is_still_neutral(self, cf: ConstitutionalFilter) -> None:
        decision = cf.evaluate(
            "send_money", FEATURES, emotion="neutral", emotion_confidence=0.9
        )
        assert decision.allow is True

    def test_confident_forbidden_emotion_is_denied(self, cf: ConstitutionalFilter) -> None:
        decision = cf.evaluate(
            "send_money", FEATURES, emotion="angry", emotion_confidence=0.56
        )
        assert decision.allow is False
        assert decision.requires_verification is False

    def test_unconfident_emotion_is_unknown_even_when_it_would_be_forbidden(
        self, cf: ConstitutionalFilter
    ) -> None:
        """Unknown emotion cannot satisfy the required list: verification, not a deny."""
        decision = cf.evaluate(
            "send_money", FEATURES, emotion="angry", emotion_confidence=0.3
        )
        assert decision.allow is False
        assert decision.requires_verification is True

    def test_unconfident_emotion_does_not_satisfy_a_required_list(
        self, cf: ConstitutionalFilter
    ) -> None:
        decision = cf.evaluate(
            "send_money", FEATURES, emotion="sincere", emotion_confidence=0.2
        )
        assert decision.allow is False
        assert decision.requires_verification is True

    def test_emotion_none_with_confidence_is_unknown(self, cf: ConstitutionalFilter) -> None:
        decision = cf.evaluate("send_money", FEATURES, emotion=None, emotion_confidence=0.9)
        assert decision.requires_verification is True


class TestThreshold:
    def test_default_minimum_is_inclusive(self, cf: ConstitutionalFilter) -> None:
        at_minimum = cf.evaluate(
            "send_money", FEATURES, emotion="angry", emotion_confidence=0.5
        )
        assert at_minimum.allow is False
        assert at_minimum.requires_verification is False  # known angry: hard deny
        just_below = cf.evaluate(
            "send_money", FEATURES, emotion="angry", emotion_confidence=0.499
        )
        assert just_below.requires_verification is True  # unknown

    def test_custom_minimum(self, cf: ConstitutionalFilter) -> None:
        strict = {"min_emotion_confidence": 0.8}
        below = cf.evaluate(
            "send_money", FEATURES, emotion="calm", emotion_confidence=0.7, **strict
        )
        assert below.requires_verification is True
        at = cf.evaluate(
            "send_money", FEATURES, emotion="calm", emotion_confidence=0.8, **strict
        )
        assert at.allow is True

    def test_lenient_minimum_accepts_what_the_caller_chooses_to_accept(
        self, cf: ConstitutionalFilter
    ) -> None:
        decision = cf.evaluate(
            "send_money",
            FEATURES,
            emotion="calm",
            emotion_confidence=0.1,
            min_emotion_confidence=0.0,
        )
        assert decision.allow is True

    def test_without_confidence_the_emotion_is_taken_as_given(
        self, cf: ConstitutionalFilter
    ) -> None:
        """Callers that pass only ``emotion`` behave as before."""
        assert cf.evaluate("send_money", FEATURES, emotion="calm").allow is True
        assert cf.evaluate("send_money", FEATURES, emotion="angry").allow is False

    @pytest.mark.parametrize("confidence", [float("nan"), float("-inf")])
    def test_unusable_confidence_is_unknown(
        self, cf: ConstitutionalFilter, confidence: float
    ) -> None:
        decision = cf.evaluate(
            "send_money", FEATURES, emotion="calm", emotion_confidence=confidence
        )
        assert decision.requires_verification is True

    @pytest.mark.parametrize("minimum", [-0.1, 1.1, float("nan"), float("inf")])
    def test_invalid_minimum_is_rejected(self, cf: ConstitutionalFilter, minimum: float) -> None:
        with pytest.raises(ValueError, match="min_emotion_confidence"):
            cf.evaluate(
                "send_money",
                FEATURES,
                emotion="calm",
                emotion_confidence=0.9,
                min_emotion_confidence=minimum,
            )

    def test_confidence_arguments_are_keyword_only(self) -> None:
        params = inspect.signature(ConstitutionalFilter.evaluate).parameters
        assert list(params)[:5] == ["self", "intent", "prosody_features", "emotion", "context"]
        for name in ("emotion_confidence", "min_emotion_confidence"):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["emotion_confidence"].default is None
        assert params["min_emotion_confidence"].default == 0.5


class TestForbiddenOnlyRules:
    """A blacklist can only fire on a known emotion."""

    RULE = ConstitutionalRule(
        name="no_anger",
        triggers=["wire"],
        forbidden_prosody=ProsodyCondition(emotion=["angry"]),
    )

    def test_known_forbidden_emotion_still_denies(self) -> None:
        decision = ConstitutionalFilter([self.RULE]).evaluate(
            "wire", FEATURES, emotion="angry", emotion_confidence=0.7
        )
        assert decision.allow is False

    def test_unknown_emotion_is_not_evidence_of_a_forbidden_one(self) -> None:
        decision = ConstitutionalFilter([self.RULE]).evaluate(
            "wire", FEATURES, emotion="angry", emotion_confidence=0.1
        )
        assert decision.allow is True

    def test_pairing_with_a_required_list_fails_closed(self) -> None:
        rule = ConstitutionalRule(
            name="calm_only",
            triggers=["wire"],
            required_prosody=ProsodyCondition(emotion=["calm"]),
            forbidden_prosody=ProsodyCondition(emotion=["angry"]),
            verification=Verification(),
        )
        decision = ConstitutionalFilter([rule]).evaluate(
            "wire", FEATURES, emotion="neutral", emotion_confidence=0.0
        )
        assert decision.requires_verification is True


class TestResolveEmotion:
    @pytest.mark.parametrize(
        ("emotion", "confidence", "minimum", "expected"),
        [
            ("Angry", None, 0.5, "angry"),
            (" calm ", 0.9, 0.5, "calm"),
            ("neutral", 0.0, 0.5, None),
            ("neutral", 0.49, 0.5, None),
            ("neutral", 0.5, 0.5, "neutral"),
            (None, 0.9, 0.5, None),
            (None, None, 0.5, None),
            ("", None, 0.5, None),
            ("   ", 0.9, 0.5, None),
            ("calm", float("nan"), 0.5, None),
        ],
    )
    def test_resolve(
        self, emotion: str | None, confidence: float | None, minimum: float, expected: str | None
    ) -> None:
        assert resolve_emotion(emotion, confidence, minimum) == expected
