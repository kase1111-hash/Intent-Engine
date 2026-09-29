"""Tests for the prosody-based rule evaluator."""

from __future__ import annotations

from prosody_protocol import SpanFeatures

from intent_engine.constitutional.evaluator import (
    PITCH_VARIANCE_THRESHOLDS,
    check_forbidden_prosody,
    check_required_prosody,
    evaluate_rule,
    match_triggers,
    most_restrictive,
)
from intent_engine.constitutional.rules import (
    ConstitutionalRule,
    ProsodyCondition,
    Verification,
)
from intent_engine.models.decision import Decision


def _make_features(
    f0_mean: float | None = None,
    f0_range: tuple[float, float] | None = None,
    speech_rate: float | None = None,
    intensity_mean: float | None = None,
) -> SpanFeatures:
    """Helper to create a SpanFeatures with sensible defaults."""
    return SpanFeatures(
        start_ms=0,
        end_ms=1000,
        text="test",
        f0_mean=f0_mean,
        f0_range=f0_range,
        speech_rate=speech_rate,
        intensity_mean=intensity_mean,
    )


class TestMatchTriggers:
    def test_exact_match(self) -> None:
        assert match_triggers("delete", ["delete", "remove"])

    def test_substring_match_trigger_in_intent(self) -> None:
        assert match_triggers("delete_all_files", ["delete"])

    def test_intent_inside_longer_trigger_is_not_a_match(self) -> None:
        # Used to match by reverse substring, which also made "end" match
        # a "send_money" trigger; only the trigger appearing in the intent
        # counts now (see test_trigger_matching.py).
        assert not match_triggers("delete", ["delete_all_files"])

    def test_case_insensitive(self) -> None:
        assert match_triggers("DELETE", ["delete"])
        assert match_triggers("delete", ["DELETE"])

    def test_no_match(self) -> None:
        assert not match_triggers("greet", ["delete", "remove"])

    def test_empty_triggers(self) -> None:
        assert not match_triggers("delete", [])

    def test_partial_word_match(self) -> None:
        assert match_triggers("send_money", ["send"])


class TestCheckRequiredProsody:
    def test_emotion_passes(self) -> None:
        cond = ProsodyCondition(emotion=["sincere", "calm"])
        features = [_make_features()]
        passed, reason = check_required_prosody(cond, features, emotion="sincere")
        assert passed is True
        assert reason is None

    def test_emotion_fails(self) -> None:
        cond = ProsodyCondition(emotion=["sincere", "calm"])
        features = [_make_features()]
        passed, reason = check_required_prosody(cond, features, emotion="angry")
        assert passed is False
        assert "Required emotion" in reason

    def test_emotion_fails_when_none(self) -> None:
        cond = ProsodyCondition(emotion=["sincere"])
        features = [_make_features()]
        passed, reason = check_required_prosody(cond, features, emotion=None)
        assert passed is False

    def test_pitch_variance_low_passes(self) -> None:
        cond = ProsodyCondition(pitch_variance="low")
        features = [_make_features(f0_range=(100.0, 120.0))]  # 20Hz spread
        passed, reason = check_required_prosody(cond, features)
        assert passed is True

    def test_pitch_variance_low_fails(self) -> None:
        cond = ProsodyCondition(pitch_variance="low")
        features = [_make_features(f0_range=(100.0, 200.0))]  # 100Hz spread
        passed, reason = check_required_prosody(cond, features)
        assert passed is False
        assert "pitch_variance" in reason

    def test_pitch_variance_high_passes(self) -> None:
        cond = ProsodyCondition(pitch_variance="high")
        features = [_make_features(f0_range=(100.0, 200.0))]  # 100Hz spread
        passed, reason = check_required_prosody(cond, features)
        assert passed is True

    def test_speaking_rate_in_range(self) -> None:
        cond = ProsodyCondition(speaking_rate=(2.0, 5.0))
        features = [_make_features(speech_rate=3.5)]
        passed, reason = check_required_prosody(cond, features)
        assert passed is True

    def test_speaking_rate_too_fast(self) -> None:
        cond = ProsodyCondition(speaking_rate=(2.0, 5.0))
        features = [_make_features(speech_rate=7.0)]
        passed, reason = check_required_prosody(cond, features)
        assert passed is False
        assert "speaking_rate" in reason

    def test_speaking_rate_too_slow(self) -> None:
        cond = ProsodyCondition(speaking_rate=(2.0, 5.0))
        features = [_make_features(speech_rate=1.0)]
        passed, reason = check_required_prosody(cond, features)
        assert passed is False

    def test_no_features_data_fails_closed(self) -> None:
        """A required measurement that could not be made is not satisfied.

        This used to assert ``passed is True`` ("checks are skipped"), which
        let silence or a failed analysis satisfy pitch/rate conditions.
        """
        cond = ProsodyCondition(pitch_variance="low", speaking_rate=(2.0, 5.0))
        features = [_make_features()]  # No f0_range or speech_rate
        passed, reason = check_required_prosody(cond, features)
        assert passed is False
        assert reason is not None
        assert "could not be measured" in reason

    def test_multiple_conditions_all_must_pass(self) -> None:
        cond = ProsodyCondition(
            emotion=["sincere"],
            pitch_variance="low",
            speaking_rate=(2.0, 5.0),
        )
        features = [_make_features(f0_range=(100.0, 120.0), speech_rate=3.0)]
        passed, reason = check_required_prosody(cond, features, emotion="sincere")
        assert passed is True

    def test_multiple_conditions_first_fails(self) -> None:
        cond = ProsodyCondition(
            emotion=["sincere"],
            pitch_variance="low",
        )
        features = [_make_features(f0_range=(100.0, 120.0))]
        passed, reason = check_required_prosody(cond, features, emotion="angry")
        assert passed is False
        assert "emotion" in reason.lower()


class TestCheckForbiddenProsody:
    def test_no_forbidden_emotion_detected(self) -> None:
        cond = ProsodyCondition(emotion=["sarcastic", "angry"])
        features = [_make_features()]
        passed, reason = check_forbidden_prosody(cond, features, emotion="sincere")
        assert passed is True

    def test_forbidden_emotion_detected(self) -> None:
        cond = ProsodyCondition(emotion=["sarcastic", "angry"])
        features = [_make_features()]
        passed, reason = check_forbidden_prosody(cond, features, emotion="sarcastic")
        assert passed is False
        assert "Forbidden emotion" in reason

    def test_no_emotion_data(self) -> None:
        cond = ProsodyCondition(emotion=["sarcastic"])
        features = [_make_features()]
        passed, reason = check_forbidden_prosody(cond, features, emotion=None)
        assert passed is True  # No emotion data = no violation


class TestEvaluateRule:
    def test_allow_when_all_checks_pass(self) -> None:
        rule = ConstitutionalRule(
            name="test",
            triggers=["test"],
            required_prosody=ProsodyCondition(emotion=["sincere"]),
        )
        features = [_make_features()]
        decision = evaluate_rule(rule, features, emotion="sincere")
        assert decision.allow is True

    def test_deny_on_forbidden_prosody(self) -> None:
        rule = ConstitutionalRule(
            name="test",
            triggers=["test"],
            forbidden_prosody=ProsodyCondition(emotion=["angry"]),
        )
        features = [_make_features()]
        decision = evaluate_rule(rule, features, emotion="angry")
        assert decision.allow is False
        assert decision.requires_verification is False
        assert "Forbidden emotion" in decision.denial_reason

    def test_deny_with_verification_on_required_prosody_fail(self) -> None:
        rule = ConstitutionalRule(
            name="test",
            triggers=["test"],
            required_prosody=ProsodyCondition(emotion=["sincere"]),
            verification=Verification(method="two_factor", retries=3),
        )
        features = [_make_features()]
        decision = evaluate_rule(rule, features, emotion="angry")
        assert decision.allow is False
        assert decision.requires_verification is True
        assert decision.verification_method == "two_factor"

    def test_deny_without_verification_on_required_prosody_fail(self) -> None:
        rule = ConstitutionalRule(
            name="test",
            triggers=["test"],
            required_prosody=ProsodyCondition(emotion=["sincere"]),
            # No verification
        )
        features = [_make_features()]
        decision = evaluate_rule(rule, features, emotion="angry")
        assert decision.allow is False
        assert decision.requires_verification is False

    def test_forbidden_takes_priority_over_required(self) -> None:
        """Forbidden check is done first, even if required would pass."""
        rule = ConstitutionalRule(
            name="test",
            triggers=["test"],
            required_prosody=ProsodyCondition(emotion=["angry"]),
            forbidden_prosody=ProsodyCondition(emotion=["angry"]),
            verification=Verification(method="explicit_confirmation"),
        )
        features = [_make_features()]
        decision = evaluate_rule(rule, features, emotion="angry")
        # Forbidden check fires first -- outright deny, no verification
        assert decision.allow is False
        assert decision.requires_verification is False

    def test_no_conditions_allows(self) -> None:
        rule = ConstitutionalRule(name="test", triggers=["test"])
        features = [_make_features()]
        decision = evaluate_rule(rule, features, emotion="angry")
        assert decision.allow is True

    def test_denial_reason_includes_rule_name(self) -> None:
        rule = ConstitutionalRule(
            name="important_rule",
            triggers=["test"],
            forbidden_prosody=ProsodyCondition(emotion=["angry"]),
        )
        features = [_make_features()]
        decision = evaluate_rule(rule, features, emotion="angry")
        assert "important_rule" in decision.denial_reason


class TestPitchVarianceThresholds:
    def test_thresholds_cover_full_range(self) -> None:
        assert "low" in PITCH_VARIANCE_THRESHOLDS
        assert "normal" in PITCH_VARIANCE_THRESHOLDS
        assert "high" in PITCH_VARIANCE_THRESHOLDS

    def test_low_starts_at_zero(self) -> None:
        lo, hi = PITCH_VARIANCE_THRESHOLDS["low"]
        assert lo == 0.0

    def test_high_is_unbounded(self) -> None:
        lo, hi = PITCH_VARIANCE_THRESHOLDS["high"]
        assert hi == float("inf")


class TestMostRestrictive:
    ALLOW = Decision(allow=True)
    CONFIRM = Decision(False, True, "explicit_confirmation", "Rule 'b': confirm")
    TWO_FACTOR = Decision(False, True, "two_factor", "Rule 'c': two factor")
    DENY = Decision(False, False, None, "Rule 'd': deny")

    def test_no_decisions_allows(self) -> None:
        assert most_restrictive([]) == self.ALLOW

    def test_ranking(self) -> None:
        ranked = [self.ALLOW, self.CONFIRM, self.TWO_FACTOR, self.DENY]
        for i, weaker in enumerate(ranked):
            for stronger in ranked[i + 1 :]:
                assert most_restrictive([weaker, stronger]) is stronger
                assert most_restrictive([stronger, weaker]) is stronger

    def test_accepts_a_generator(self) -> None:
        assert most_restrictive(d for d in (self.CONFIRM, self.DENY)) is self.DENY

    def test_equal_severity_is_resolved_independently_of_order(self) -> None:
        first = Decision(False, False, None, "Rule 'a': deny")
        second = Decision(False, False, None, "Rule 'z': deny")
        assert most_restrictive([second, first]) is first
        assert most_restrictive([first, second]) is first

    def test_unknown_verification_method_ranks_as_the_strictest_verification(self) -> None:
        odd = Decision(False, True, "carrier_pigeon", "Rule 'e': odd")
        assert most_restrictive([self.CONFIRM, odd]) is odd
        assert most_restrictive([odd, self.CONFIRM]) is odd
        assert most_restrictive([self.DENY, odd]) is self.DENY
