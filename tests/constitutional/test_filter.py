"""Tests for the ConstitutionalFilter class."""

from __future__ import annotations

from pathlib import Path

import pytest
from prosody_protocol import SpanFeatures

from intent_engine.constitutional import ConstitutionalFilter
from intent_engine.constitutional.rules import (
    ConstitutionalRule,
    ProsodyCondition,
    Verification,
)
from intent_engine.models.decision import Decision

SAMPLE_RULES_PATH = Path(__file__).parent / "sample_rules.yaml"


def _make_features(
    f0_range: tuple[float, float] | None = None,
    speech_rate: float | None = None,
) -> SpanFeatures:
    return SpanFeatures(
        start_ms=0,
        end_ms=1000,
        text="test",
        f0_range=f0_range,
        speech_rate=speech_rate,
    )


class TestConstitutionalFilterConstruction:
    def test_from_rules_list(self) -> None:
        rules = [
            ConstitutionalRule(name="r1", triggers=["delete"]),
            ConstitutionalRule(name="r2", triggers=["transfer"]),
        ]
        cf = ConstitutionalFilter(rules)
        assert len(cf.rules) == 2

    def test_from_yaml(self) -> None:
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        assert len(cf.rules) == 4

    def test_rules_returns_copy(self) -> None:
        rules = [ConstitutionalRule(name="r1", triggers=["delete"])]
        cf = ConstitutionalFilter(rules)
        returned = cf.rules
        returned.append(ConstitutionalRule(name="extra"))
        assert len(cf.rules) == 1  # Original not modified

    def test_from_yaml_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            ConstitutionalFilter.from_yaml("/nonexistent/rules.yaml")


class TestConstitutionalFilterEvaluate:
    """Tests covering all four decision branches from the spec."""

    def test_no_matching_rules_allows(self) -> None:
        """Branch 1: No rules match intent -> allow."""
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        features = [_make_features()]
        decision = cf.evaluate("greet_user", features, emotion="joyful")
        assert decision.allow is True

    def test_matching_rules_prosody_passes_allows(self) -> None:
        """Branch 2: Rules match AND prosody passes -> allow."""
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        features = [_make_features(f0_range=(100.0, 115.0), speech_rate=3.5)]
        decision = cf.evaluate("delete_files", features, emotion="sincere")
        assert decision.allow is True

    def test_matching_rules_prosody_fails_with_verification(self) -> None:
        """Branch 3: Rules match AND prosody fails AND verification defined."""
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        features = [_make_features(f0_range=(100.0, 115.0), speech_rate=3.5)]
        decision = cf.evaluate("delete_files", features, emotion="uncertain")
        assert decision.allow is False
        assert decision.requires_verification is True
        assert decision.verification_method == "explicit_confirmation"
        assert decision.denial_reason is not None

    def test_matching_rules_prosody_fails_no_verification(self) -> None:
        """Branch 4: Rules match AND prosody fails AND no verification."""
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        features = [_make_features()]
        decision = cf.evaluate("delete_account", features, emotion="angry")
        # account_deletion has forbidden_prosody: [sarcastic, angry, frustrated]
        assert decision.allow is False
        assert decision.requires_verification is False

    def test_forbidden_emotion_blocks_regardless(self) -> None:
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        features = [_make_features(f0_range=(100.0, 115.0), speech_rate=3.5)]
        decision = cf.evaluate("delete_files", features, emotion="sarcastic")
        assert decision.allow is False
        assert decision.requires_verification is False
        assert "Forbidden emotion" in decision.denial_reason

    def test_financial_transaction_two_factor(self) -> None:
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        features = [_make_features()]
        # Emotion that doesn't match required [sincere, calm, neutral]
        decision = cf.evaluate("send_money", features, emotion="uncertain")
        assert decision.allow is False
        assert decision.requires_verification is True
        assert decision.verification_method == "two_factor"

    def test_financial_transaction_fearful_blocked(self) -> None:
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        features = [_make_features()]
        decision = cf.evaluate("transfer", features, emotion="fearful")
        assert decision.allow is False
        assert decision.requires_verification is False
        assert "Forbidden" in decision.denial_reason

    def test_emergency_with_fast_speech_allowed(self) -> None:
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        features = [_make_features(speech_rate=6.0)]
        decision = cf.evaluate("emergency", features, emotion="fearful")
        assert decision.allow is True

    def test_emergency_with_slow_speech_needs_verification(self) -> None:
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        features = [_make_features(speech_rate=2.0)]
        decision = cf.evaluate("emergency", features, emotion="neutral")
        assert decision.allow is False
        assert decision.requires_verification is True
        assert decision.verification_method == "explicit_confirmation"

    def test_emergency_sarcastic_blocked(self) -> None:
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        features = [_make_features(speech_rate=6.0)]
        decision = cf.evaluate("emergency", features, emotion="sarcastic")
        assert decision.allow is False
        assert decision.requires_verification is False

    def test_context_param_accepted(self) -> None:
        """Context is accepted but not currently used in evaluation."""
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        features = [_make_features()]
        decision = cf.evaluate(
            "greet_user",
            features,
            emotion="neutral",
            context={"user_id": "123", "session_risk": "low"},
        )
        assert decision.allow is True


class TestConstitutionalFilterMultipleRules:
    def test_most_restrictive_wins(self) -> None:
        """When multiple rules match, the most restrictive decision wins."""
        rules = [
            ConstitutionalRule(
                name="strict",
                triggers=["action"],
                forbidden_prosody=ProsodyCondition(emotion=["angry"]),
            ),
            ConstitutionalRule(
                name="lenient",
                triggers=["action"],
                required_prosody=ProsodyCondition(emotion=["sincere", "calm", "neutral"]),
                verification=Verification(method="explicit_confirmation"),
            ),
        ]
        cf = ConstitutionalFilter(rules)
        features = [_make_features()]
        # "angry" hits the forbidden in the strict rule
        decision = cf.evaluate("action", features, emotion="angry")
        assert decision.allow is False
        assert decision.requires_verification is False

    def test_all_rules_pass_allows(self) -> None:
        rules = [
            ConstitutionalRule(
                name="r1",
                triggers=["action"],
                required_prosody=ProsodyCondition(emotion=["sincere"]),
            ),
            ConstitutionalRule(
                name="r2",
                triggers=["action"],
                required_prosody=ProsodyCondition(emotion=["sincere"]),
            ),
        ]
        cf = ConstitutionalFilter(rules)
        features = [_make_features()]
        decision = cf.evaluate("action", features, emotion="sincere")
        assert decision.allow is True


class TestConstitutionalFilterEdgeCases:
    def test_empty_features_list(self) -> None:
        rules = [
            ConstitutionalRule(
                name="test",
                triggers=["test"],
                required_prosody=ProsodyCondition(emotion=["sincere"]),
            ),
        ]
        cf = ConstitutionalFilter(rules)
        decision = cf.evaluate("test", [], emotion="sincere")
        assert decision.allow is True

    def test_empty_rules(self) -> None:
        cf = ConstitutionalFilter([])
        features = [_make_features()]
        decision = cf.evaluate("anything", features, emotion="angry")
        assert decision.allow is True

    def test_decision_is_decision_type(self) -> None:
        cf = ConstitutionalFilter([])
        features = [_make_features()]
        decision = cf.evaluate("test", features)
        assert isinstance(decision, Decision)
