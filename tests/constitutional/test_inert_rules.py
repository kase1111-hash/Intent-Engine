"""Rules that could never restrict anything are refused or reported.

A ``verification`` block only applies after ``required_prosody`` fails, so one
written without it is dead configuration that reads like a safeguard; an empty
``ProsodyCondition`` would check nothing.  Both are refused when the rule is
built (from YAML or in code).  A rule with neither prosody section is a valid,
tested shape (a trigger that is always allowed), so it loads, with a warning.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from intent_engine.constitutional import ConstitutionalFilter
from intent_engine.constitutional.rules import (
    ConstitutionalRule,
    ProsodyCondition,
    Verification,
    parse_rules_yaml,
)

RULES_LOGGER = "intent_engine.constitutional.rules"


def _load(tmp_path: Path, body: str) -> list[ConstitutionalRule]:
    path = tmp_path / "rules.yaml"
    path.write_text(f"rules:\n  wire:\n    triggers: [wire money]\n    {body}\n", encoding="utf-8")
    return parse_rules_yaml(path)


class TestVerificationNeedsRequiredProsody:
    @pytest.mark.parametrize(
        "body",
        [
            "verification: {method: two_factor}",
            "verification: {retries: 2}",
            "forbidden_prosody: {emotion: [angry]}\n    verification: {method: two_factor}",
        ],
    )
    def test_yaml_rule_is_rejected(self, tmp_path: Path, body: str) -> None:
        with pytest.raises(ValueError, match="required_prosody") as excinfo:
            _load(tmp_path, body)
        assert "wire" in str(excinfo.value)
        assert "verification" in str(excinfo.value)

    def test_constructor_rule_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="required_prosody"):
            ConstitutionalRule(
                name="wire", triggers=["wire money"], verification=Verification("two_factor")
            )

    def test_forbidden_only_with_verification_is_rejected_by_the_constructor(self) -> None:
        with pytest.raises(ValueError, match="required_prosody"):
            ConstitutionalRule(
                name="wire",
                triggers=["wire money"],
                forbidden_prosody=ProsodyCondition(emotion=["angry"]),
                verification=Verification(),
            )

    def test_required_prosody_with_verification_still_works(self, tmp_path: Path) -> None:
        rules = _load(
            tmp_path,
            "required_prosody: {emotion: [calm]}\n    verification: {method: two_factor}",
        )
        decision = ConstitutionalFilter(rules).evaluate("wire_money", [], emotion=None)
        assert decision.requires_verification
        assert decision.verification_method == "two_factor"


class TestEmptyCondition:
    @pytest.mark.parametrize("kwargs", [{}, {"emotion": []}, {"emotion": ()}])
    def test_a_condition_that_checks_nothing_is_rejected(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ProsodyCondition(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("section", ["required_prosody", "forbidden_prosody"])
    def test_rule_sections_cannot_be_empty_in_code(self, section: str) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ConstitutionalRule(name="r", triggers=["x"], **{section: ProsodyCondition()})

    def test_the_verification_scenario_of_an_empty_required_condition(self) -> None:
        # Previously loaded fine and evaluated to allow for every caller.
        with pytest.raises(ValueError, match="at least one"):
            ConstitutionalRule(
                name="a",
                triggers=["wire money"],
                required_prosody=ProsodyCondition(),
                verification=Verification("two_factor"),
            )

    def test_any_single_field_is_enough(self) -> None:
        ProsodyCondition(emotion=["calm"])
        ProsodyCondition(pitch_variance="low")
        ProsodyCondition(speaking_rate=(2.0, 5.0))


class TestRulesWithoutConditions:
    def test_a_trigger_only_rule_is_valid_and_allows(self, tmp_path: Path) -> None:
        rules = _load(tmp_path, "")

        assert rules[0].required_prosody is None
        assert rules[0].forbidden_prosody is None
        assert ConstitutionalFilter(rules).evaluate("wire_money", []).allow

    def test_loading_one_warns_that_it_never_restricts(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=RULES_LOGGER):
            _load(tmp_path, "")

        messages = [r.getMessage() for r in caplog.records if r.name == RULES_LOGGER]
        assert any("wire" in m and "never restrict" in m for m in messages), messages

    def test_a_rule_with_a_condition_does_not_warn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=RULES_LOGGER):
            _load(tmp_path, "forbidden_prosody: {emotion: [angry]}")

        assert not [r for r in caplog.records if "never restrict" in r.getMessage()]
