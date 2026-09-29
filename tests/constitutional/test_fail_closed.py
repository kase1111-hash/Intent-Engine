"""A rule that needs a measurement must not pass when nothing was measured (audit #37).

Silence, unvoiced spans, a failed analysis or a recognizer that reports no
speech rate used to satisfy ``pitch_variance`` and ``speaking_rate``
conditions, turning "measured, calm speech" into an unconditional allow.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from prosody_protocol import SpanFeatures

from intent_engine.constitutional import ConstitutionalFilter
from intent_engine.constitutional.evaluator import check_required_prosody, evaluate_rule
from intent_engine.constitutional.rules import (
    ConstitutionalRule,
    ProsodyCondition,
    Verification,
)

SAMPLE_RULES_PATH = Path(__file__).parent / "sample_rules.yaml"


def _span(**kwargs: object) -> SpanFeatures:
    return SpanFeatures(start_ms=0, end_ms=500, text="w", **kwargs)  # type: ignore[arg-type]


def _unmeasured_variants() -> list[tuple[str, list[SpanFeatures]]]:
    return [
        ("no spans", []),
        ("all None", [_span()]),
        ("several all-None spans", [_span(), _span()]),
        ("only f0_mean", [_span(f0_mean=120.0, intensity_mean=60.0)]),
    ]


class TestSampleRulesFailClosed:
    @pytest.mark.parametrize(("label", "features"), _unmeasured_variants())
    def test_emergency_needs_a_measured_rate(
        self, label: str, features: list[SpanFeatures]
    ) -> None:
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        decision = cf.evaluate("emergency", features, emotion=None)
        assert decision.allow is False, label
        assert decision.requires_verification is True
        assert decision.verification_method == "explicit_confirmation"
        assert decision.denial_reason is not None
        assert "could not be measured" in decision.denial_reason

    @pytest.mark.parametrize(("label", "features"), _unmeasured_variants())
    def test_delete_files_needs_measured_pitch_and_rate(
        self, label: str, features: list[SpanFeatures]
    ) -> None:
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        decision = cf.evaluate("delete_files", features, emotion="sincere")
        assert decision.allow is False, label
        assert decision.requires_verification is True

    def test_measured_speech_still_passes(self) -> None:
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        features = [_span(f0_range=(100.0, 110.0), speech_rate=3.5)]
        assert cf.evaluate("delete_files", features, emotion="sincere").allow is True


class TestRuleFallback:
    def test_hard_deny_when_the_rule_has_no_verification(self) -> None:
        rule = ConstitutionalRule(
            name="strict",
            triggers=["wire"],
            required_prosody=ProsodyCondition(speaking_rate=(2.0, 5.0)),
        )
        decision = evaluate_rule(rule, [_span()], emotion=None)
        assert decision.allow is False
        assert decision.requires_verification is False

    def test_verification_when_the_rule_has_one(self) -> None:
        rule = ConstitutionalRule(
            name="lenient",
            triggers=["wire"],
            required_prosody=ProsodyCondition(pitch_variance="low"),
            verification=Verification(method="two_factor"),
        )
        decision = evaluate_rule(rule, [], emotion=None)
        assert decision.requires_verification is True
        assert decision.verification_method == "two_factor"

    def test_one_missing_measurement_is_enough_to_fail(self) -> None:
        rule = ConstitutionalRule(
            name="both",
            triggers=["wire"],
            required_prosody=ProsodyCondition(pitch_variance="low", speaking_rate=(2.0, 5.0)),
        )
        only_rate = [_span(speech_rate=3.0)]
        assert evaluate_rule(rule, only_rate, emotion=None).allow is False
        only_pitch = [_span(f0_range=(100.0, 105.0))]
        assert evaluate_rule(rule, only_pitch, emotion=None).allow is False
        both = [_span(f0_range=(100.0, 105.0), speech_rate=3.0)]
        assert evaluate_rule(rule, both, emotion=None).allow is True


class TestWhatCountsAsMeasured:
    @pytest.mark.parametrize("rate", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_rate_is_not_a_measurement(self, rate: float) -> None:
        cond = ProsodyCondition(speaking_rate=(0.0, 100.0))
        passed, reason = check_required_prosody(cond, [_span(speech_rate=rate)])
        assert passed is False
        assert reason is not None
        assert "could not be measured" in reason

    @pytest.mark.parametrize(
        "f0_range",
        [(float("nan"), 120.0), (0.0, 120.0), (-5.0, 120.0), (200.0, 100.0)],
    )
    def test_unusable_pitch_range_is_not_a_measurement(
        self, f0_range: tuple[float, float]
    ) -> None:
        cond = ProsodyCondition(pitch_variance="low")
        passed, reason = check_required_prosody(cond, [_span(f0_range=f0_range)])
        assert passed is False
        assert reason is not None
        assert "could not be measured" in reason

    def test_partial_measurements_use_the_spans_that_have_them(self) -> None:
        """Unvoiced words are skipped; the voiced ones decide."""
        cond = ProsodyCondition(speaking_rate=(2.0, 5.0))
        features = [_span(), _span(speech_rate=3.0), _span(speech_rate=4.0)]
        passed, _ = check_required_prosody(cond, features)
        assert passed is True


class TestConditionsThatNeedNoMeasurement:
    def test_emotion_only_rule_is_decided_by_the_emotion(self) -> None:
        """Emotion is supplied by the caller, not measured from the spans."""
        rule = ConstitutionalRule(
            name="emotion_only",
            triggers=["test"],
            required_prosody=ProsodyCondition(emotion=["sincere"]),
        )
        assert evaluate_rule(rule, [], emotion="sincere").allow is True
        assert evaluate_rule(rule, [], emotion=None).allow is False

    def test_forbidden_only_rule_needs_no_measurement(self) -> None:
        rule = ConstitutionalRule(
            name="forbid",
            triggers=["test"],
            forbidden_prosody=ProsodyCondition(emotion=["angry"]),
        )
        assert evaluate_rule(rule, [], emotion=None).allow is True
        assert evaluate_rule(rule, [], emotion="angry").allow is False


class TestUnvalidatedConditions:
    def test_unknown_pitch_level_never_passes(self) -> None:
        """Conditions are validated on construction; if one gets past that, it fails closed."""
        cond = ProsodyCondition(pitch_variance="low")
        object.__setattr__(cond, "pitch_variance", "hihg")
        passed, reason = check_required_prosody(cond, [_span(f0_range=(100.0, 105.0))])
        assert passed is False
        assert reason is not None
        assert "Unknown pitch_variance" in reason
