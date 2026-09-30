"""When several rules match, the most restrictive decision wins (audit #39).

hard deny > two_factor > explicit_confirmation > allow, whatever the order
of the rules.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from prosody_protocol import SpanFeatures

from intent_engine.constitutional import ConstitutionalFilter
from intent_engine.constitutional.rules import (
    ConstitutionalRule,
    ProsodyCondition,
    Verification,
    parse_rules_yaml,
)
from intent_engine.models.decision import Decision

SAMPLE_RULES_PATH = Path(__file__).parent / "sample_rules.yaml"
SAMPLE_RULES = parse_rules_yaml(SAMPLE_RULES_PATH)
CALM_SPEECH = [SpanFeatures(0, 500, "w", f0_range=(100.0, 110.0), speech_rate=3.0)]


def _permutations() -> list[tuple[ConstitutionalRule, ...]]:
    return list(itertools.permutations(SAMPLE_RULES))


ALL_KINDS = ["hard", "two_factor", "confirm", "open"]
VERIFYING_KINDS = ["two_factor", "confirm", "open"]


class TestSampleRulesInAnyOrder:
    """``delete_account`` matches both ``delete_files`` and ``account_deletion``."""

    def test_the_sample_intent_really_matches_two_rules(self) -> None:
        from intent_engine.constitutional.evaluator import match_triggers

        matching = [r.name for r in SAMPLE_RULES if match_triggers("delete_account", r.triggers)]
        assert sorted(matching) == ["account_deletion", "delete_files"]

    @pytest.mark.parametrize("order", _permutations())
    def test_hard_deny_beats_verification_regardless_of_order(
        self, order: tuple[ConstitutionalRule, ...]
    ) -> None:
        """account_deletion forbids frustrated and has no verification fallback;
        delete_files would only ask for a confirmation."""
        decision = ConstitutionalFilter(list(order)).evaluate(
            "delete_account", CALM_SPEECH, emotion="frustrated"
        )
        assert decision.allow is False
        assert decision.requires_verification is False
        assert decision.verification_method is None
        assert decision.denial_reason is not None
        assert "account_deletion" in decision.denial_reason

    def test_the_decision_is_identical_for_every_order(self) -> None:
        decisions = {
            ConstitutionalFilter(list(order)).evaluate(
                "delete_account", CALM_SPEECH, emotion="uncertain"
            )
            for order in _permutations()
        }
        assert len(decisions) == 1

    @pytest.mark.parametrize("order", _permutations())
    def test_allow_only_when_every_matching_rule_allows(
        self, order: tuple[ConstitutionalRule, ...]
    ) -> None:
        cf = ConstitutionalFilter(list(order))
        assert cf.evaluate("delete_account", CALM_SPEECH, emotion="sincere").allow is True
        # sincere is fine for account_deletion but delete_files also needs
        # measured, low-pitch, moderate-rate speech.
        rushed = [SpanFeatures(0, 500, "w", f0_range=(100.0, 110.0), speech_rate=9.0)]
        decision = cf.evaluate("delete_account", rushed, emotion="sincere")
        assert decision.allow is False
        assert decision.requires_verification is True


class TestSeverityOrder:
    @staticmethod
    def _rules() -> dict[str, ConstitutionalRule]:
        need_calm = ProsodyCondition(emotion=["calm"])
        return {
            "hard": ConstitutionalRule(name="hard", triggers=["pay"], required_prosody=need_calm),
            "two_factor": ConstitutionalRule(
                name="two_factor",
                triggers=["pay"],
                required_prosody=need_calm,
                verification=Verification(method="two_factor"),
            ),
            "confirm": ConstitutionalRule(
                name="confirm",
                triggers=["pay"],
                required_prosody=need_calm,
                verification=Verification(method="explicit_confirmation"),
            ),
            "open": ConstitutionalRule(name="open", triggers=["pay"]),
        }

    @pytest.mark.parametrize("names", list(itertools.permutations(ALL_KINDS)))
    def test_hard_deny_wins(self, names: tuple[str, ...]) -> None:
        rules = self._rules()
        decision = ConstitutionalFilter([rules[n] for n in names]).evaluate(
            "pay", CALM_SPEECH, emotion="sad"
        )
        assert decision.allow is False
        assert decision.requires_verification is False
        assert decision.denial_reason is not None
        assert "'hard'" in decision.denial_reason

    @pytest.mark.parametrize("names", list(itertools.permutations(VERIFYING_KINDS)))
    def test_two_factor_beats_explicit_confirmation(self, names: tuple[str, ...]) -> None:
        rules = self._rules()
        decision = ConstitutionalFilter([rules[n] for n in names]).evaluate(
            "pay", CALM_SPEECH, emotion="sad"
        )
        assert decision.requires_verification is True
        assert decision.verification_method == "two_factor"

    def test_explicit_confirmation_beats_allow(self) -> None:
        rules = self._rules()
        for names in (("open", "confirm"), ("confirm", "open")):
            decision = ConstitutionalFilter([rules[n] for n in names]).evaluate(
                "pay", CALM_SPEECH, emotion="sad"
            )
            assert decision.verification_method == "explicit_confirmation"

    def test_ties_are_broken_by_rule_name_not_position(self) -> None:
        forbid_a = ConstitutionalRule(
            name="a_rule", triggers=["x"], forbidden_prosody=ProsodyCondition(emotion=["angry"])
        )
        forbid_b = ConstitutionalRule(
            name="b_rule", triggers=["x"], forbidden_prosody=ProsodyCondition(emotion=["angry"])
        )
        first = ConstitutionalFilter([forbid_a, forbid_b]).evaluate("x", [], emotion="angry")
        second = ConstitutionalFilter([forbid_b, forbid_a]).evaluate("x", [], emotion="angry")
        assert first == second
        assert isinstance(first, Decision)
        assert first.denial_reason is not None
        assert "a_rule" in first.denial_reason
