"""Tests for the YAML rule schema and parser."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from intent_engine.constitutional.rules import (
    ConstitutionalRule,
    ProsodyCondition,
    Verification,
    parse_rules_yaml,
)

SAMPLE_RULES_PATH = Path(__file__).parent / "sample_rules.yaml"


class TestProsodyCondition:
    def test_default_values(self) -> None:
        cond = ProsodyCondition()
        assert cond.emotion == []
        assert cond.pitch_variance is None
        assert cond.speaking_rate is None

    def test_with_values(self) -> None:
        cond = ProsodyCondition(
            emotion=["sincere", "calm"],
            pitch_variance="low",
            speaking_rate=(2.0, 5.0),
        )
        assert cond.emotion == ["sincere", "calm"]
        assert cond.pitch_variance == "low"
        assert cond.speaking_rate == (2.0, 5.0)

    def test_frozen(self) -> None:
        cond = ProsodyCondition()
        with pytest.raises(AttributeError):
            cond.pitch_variance = "high"  # type: ignore[misc]


class TestVerification:
    def test_default_values(self) -> None:
        v = Verification()
        assert v.method == "explicit_confirmation"
        assert v.retries == 1

    def test_custom_values(self) -> None:
        v = Verification(method="two_factor", retries=3)
        assert v.method == "two_factor"
        assert v.retries == 3

    def test_frozen(self) -> None:
        v = Verification()
        with pytest.raises(AttributeError):
            v.method = "other"  # type: ignore[misc]


class TestConstitutionalRule:
    def test_minimal_rule(self) -> None:
        rule = ConstitutionalRule(name="test")
        assert rule.name == "test"
        assert rule.triggers == []
        assert rule.required_prosody is None
        assert rule.forbidden_prosody is None
        assert rule.verification is None

    def test_full_rule(self) -> None:
        rule = ConstitutionalRule(
            name="delete_files",
            triggers=["delete", "remove"],
            required_prosody=ProsodyCondition(emotion=["sincere"]),
            forbidden_prosody=ProsodyCondition(emotion=["sarcastic"]),
            verification=Verification(method="two_factor", retries=2),
        )
        assert rule.name == "delete_files"
        assert len(rule.triggers) == 2
        assert rule.required_prosody is not None
        assert rule.forbidden_prosody is not None
        assert rule.verification is not None


class TestParseRulesYaml:
    def test_parse_sample_rules(self) -> None:
        rules = parse_rules_yaml(SAMPLE_RULES_PATH)
        assert len(rules) == 4

    def test_rule_names(self) -> None:
        rules = parse_rules_yaml(SAMPLE_RULES_PATH)
        names = [r.name for r in rules]
        assert "delete_files" in names
        assert "financial_transaction" in names
        assert "account_deletion" in names
        assert "emergency_action" in names

    def test_delete_files_rule(self) -> None:
        rules = parse_rules_yaml(SAMPLE_RULES_PATH)
        rule = next(r for r in rules if r.name == "delete_files")

        assert "delete" in rule.triggers
        assert "remove" in rule.triggers
        assert "erase" in rule.triggers

        assert rule.required_prosody is not None
        assert "sincere" in rule.required_prosody.emotion
        assert rule.required_prosody.pitch_variance == "low"
        assert rule.required_prosody.speaking_rate == (2.0, 5.0)

        assert rule.forbidden_prosody is not None
        assert "sarcastic" in rule.forbidden_prosody.emotion
        assert "angry" in rule.forbidden_prosody.emotion

        assert rule.verification is not None
        assert rule.verification.method == "explicit_confirmation"
        assert rule.verification.retries == 2

    def test_account_deletion_has_no_verification(self) -> None:
        rules = parse_rules_yaml(SAMPLE_RULES_PATH)
        rule = next(r for r in rules if r.name == "account_deletion")
        assert rule.verification is None

    def test_financial_transaction_uses_two_factor(self) -> None:
        rules = parse_rules_yaml(SAMPLE_RULES_PATH)
        rule = next(r for r in rules if r.name == "financial_transaction")
        assert rule.verification is not None
        assert rule.verification.method == "two_factor"
        assert rule.verification.retries == 3

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="Rules file not found"):
            parse_rules_yaml("/nonexistent/path/rules.yaml")

    def test_invalid_yaml_no_rules_key(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("something_else: true\n")
            f.flush()
            with pytest.raises(ValueError, match="expected a top-level 'rules' key"):
                parse_rules_yaml(f.name)
        Path(f.name).unlink()

    def test_invalid_yaml_rules_not_mapping(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("rules:\n  - item1\n  - item2\n")
            f.flush()
            with pytest.raises(ValueError, match="'rules' must be a mapping"):
                parse_rules_yaml(f.name)
        Path(f.name).unlink()

    def test_invalid_rule_not_mapping(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("rules:\n  bad_rule: not_a_mapping\n")
            f.flush()
            with pytest.raises(ValueError, match="expected a mapping"):
                parse_rules_yaml(f.name)
        Path(f.name).unlink()

    def test_minimal_rule_yaml(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("rules:\n  simple:\n    triggers:\n      - test\n")
            f.flush()
            rules = parse_rules_yaml(f.name)
            assert len(rules) == 1
            assert rules[0].name == "simple"
            assert rules[0].triggers == ["test"]
            assert rules[0].required_prosody is None
            assert rules[0].forbidden_prosody is None
            assert rules[0].verification is None
        Path(f.name).unlink()
