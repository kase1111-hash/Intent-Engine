"""Strict rule schema: nothing in a rules file may be silently dropped.

Covers audit #40 (malformed/unknown conditions used to fail open), #47
(encoding and YAML errors) and #48 (mutability and emotion case).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from prosody_protocol import SpanFeatures

from intent_engine.constitutional import ConstitutionalFilter
from intent_engine.constitutional.evaluator import (
    check_forbidden_prosody,
    check_required_prosody,
)
from intent_engine.constitutional.rules import (
    ConstitutionalRule,
    ProsodyCondition,
    Verification,
    parse_rules_yaml,
)

SAMPLE_RULES_PATH = Path(__file__).parent / "sample_rules.yaml"


def _load(tmp_path: Path, text: str) -> list[ConstitutionalRule]:
    path = tmp_path / "rules.yaml"
    path.write_text(text, encoding="utf-8")
    return parse_rules_yaml(path)


def _rule(body: str) -> str:
    """A one-rule file named ``myrule`` with *body* indented under it."""
    return f"rules:\n  myrule:\n    triggers: [delete]\n    {body}\n"


class TestUnknownAndMalformedContent:
    @pytest.mark.parametrize(
        ("body", "needle"),
        [
            # Vocabulary from before the a3 semantics, or plain typos.
            ("required_prosody: {quality: [modal]}", "quality"),
            ("required_prosody: {jitter: [0.0, 2.0]}", "jitter"),
            ("required_prosody: {speaking_rates: [2, 5]}", "speaking_rates"),
            ("required_prosodyy: {emotion: [calm]}", "required_prosodyy"),
            ("forbidden_prosody: {emotions: [angry]}", "emotions"),
            ("verification: {level: 2}", "level"),
            ("severity: high", "severity"),
        ],
    )
    def test_unknown_keys_are_rejected(
        self, tmp_path: Path, body: str, needle: str
    ) -> None:
        with pytest.raises(ValueError, match=needle) as excinfo:
            _load(tmp_path, _rule(body))
        assert "myrule" in str(excinfo.value)

    @pytest.mark.parametrize(
        "body",
        [
            "required_prosody: {speaking_rate: 4}",
            "required_prosody: {speaking_rate: [4]}",
            "required_prosody: {speaking_rate: [1, 2, 3]}",
            "required_prosody: {speaking_rate: [5, 2]}",
            "required_prosody: {speaking_rate: [a, b]}",
            "required_prosody: {speaking_rate: [true, 5]}",
            "required_prosody: {speaking_rate: [.nan, 5]}",
            "required_prosody: {speaking_rate: [-1, 5]}",
            "required_prosody: {speaking_rate: }",
        ],
    )
    def test_speaking_rate_must_be_a_min_max_pair(
        self, tmp_path: Path, body: str
    ) -> None:
        with pytest.raises(ValueError, match="speaking_rate") as excinfo:
            _load(tmp_path, _rule(body))
        assert "myrule" in str(excinfo.value)

    @pytest.mark.parametrize("label", ["hihg", "medium", "5", "true", "''"])
    def test_unknown_pitch_variance_is_rejected(
        self, tmp_path: Path, label: str
    ) -> None:
        with pytest.raises(ValueError, match="pitch_variance") as excinfo:
            _load(tmp_path, _rule(f"required_prosody: {{pitch_variance: {label}}}"))
        assert "myrule" in str(excinfo.value)

    def test_blank_pitch_variance_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="pitch_variance") as excinfo:
            _load(tmp_path, _rule("required_prosody: {pitch_variance: }"))
        assert "myrule" in str(excinfo.value)

    def test_pitch_variance_is_case_insensitive(self, tmp_path: Path) -> None:
        rules = _load(tmp_path, _rule("required_prosody: {pitch_variance: Low}"))
        assert rules[0].required_prosody is not None
        assert rules[0].required_prosody.pitch_variance == "low"

    @pytest.mark.parametrize(
        "body",
        [
            "forbidden_prosody: {speaking_rate: [8, 20]}",
            "forbidden_prosody: {pitch_variance: high}",
        ],
    )
    def test_forbidden_prosody_only_supports_emotion(
        self, tmp_path: Path, body: str
    ) -> None:
        with pytest.raises(ValueError, match="only 'emotion'") as excinfo:
            _load(tmp_path, _rule(body))
        assert "myrule" in str(excinfo.value)

    @pytest.mark.parametrize(
        "emotion",
        ["sincere", "", "[]", "[calm, 42]", "[[calm]]", "['']", "[null]", "[yes]"],
    )
    def test_emotion_must_be_a_list_of_labels(
        self, tmp_path: Path, emotion: str
    ) -> None:
        with pytest.raises(ValueError, match="emotion") as excinfo:
            _load(tmp_path, _rule(f"required_prosody: {{emotion: {emotion}}}"))
        assert "myrule" in str(excinfo.value)

    @pytest.mark.parametrize(
        "triggers",
        ["delete", "", "[]", "['']", "['  ']", "['___']", "[no, 42]", "[[a]]", "{a: b}"],
    )
    def test_triggers_must_be_a_non_empty_list_of_phrases(
        self, tmp_path: Path, triggers: str
    ) -> None:
        text = (
            f"rules:\n  myrule:\n    triggers: {triggers}\n"
            "    required_prosody: {emotion: [calm]}\n"
        )
        with pytest.raises(ValueError, match="triggers") as excinfo:
            _load(tmp_path, text)
        assert "myrule" in str(excinfo.value)

    def test_missing_triggers_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="triggers") as excinfo:
            _load(tmp_path, "rules:\n  myrule:\n    required_prosody: {emotion: [calm]}\n")
        assert "myrule" in str(excinfo.value)

    @pytest.mark.parametrize(
        "body",
        [
            "required_prosody: [calm]",
            "required_prosody: calm",
            "forbidden_prosody: [angry]",
            # A section that says nothing would silently become "no condition".
            "required_prosody:",
            "required_prosody: {}",
            "forbidden_prosody:",
            "forbidden_prosody: {}",
            "verification:",
            "verification: {}",
            "verification: true",
            "verification: two_factor",
        ],
    )
    def test_sections_must_be_non_empty_mappings(
        self, tmp_path: Path, body: str
    ) -> None:
        with pytest.raises(ValueError, match="mapping") as excinfo:
            _load(tmp_path, _rule(body))
        assert "myrule" in str(excinfo.value)

    @pytest.mark.parametrize(
        ("verification", "needle"),
        [
            ("{method: bogus_method}", "method"),
            ("{method: 2fa}", "method"),
            ("{method: [two_factor]}", "method"),
            ("{retries: -1}", "retries"),
            ("{retries: 2.5}", "retries"),
            ("{retries: '2'}", "retries"),
            ("{retries: true}", "retries"),
        ],
    )
    def test_verification_is_validated(
        self, tmp_path: Path, verification: str, needle: str
    ) -> None:
        with pytest.raises(ValueError, match=needle) as excinfo:
            _load(
                tmp_path,
                _rule(f"required_prosody: {{emotion: [calm]}}\n    verification: {verification}"),
            )
        assert "myrule" in str(excinfo.value)

    def test_verification_defaults_are_kept(self, tmp_path: Path) -> None:
        rules = _load(
            tmp_path,
            _rule("required_prosody: {emotion: [calm]}\n    verification: {retries: 0}"),
        )
        assert rules[0].verification == Verification("explicit_confirmation", 0)

    def test_duplicate_rule_names_are_rejected(self, tmp_path: Path) -> None:
        text = (
            "rules:\n"
            "  delete_files:\n    triggers: [delete]\n    forbidden_prosody: {emotion: [angry]}\n"
            "  delete_files:\n    triggers: [remove]\n    required_prosody: {emotion: [calm]}\n"
        )
        with pytest.raises(ValueError, match="duplicate") as excinfo:
            _load(tmp_path, text)
        assert "delete_files" in str(excinfo.value)

    def test_empty_rule_set_is_rejected(self, tmp_path: Path) -> None:
        """An empty rule set would allow every action."""
        with pytest.raises(ValueError, match="empty"):
            _load(tmp_path, "rules: {}\n")

    def test_unknown_top_level_key_is_rejected(self, tmp_path: Path) -> None:
        text = "rules:\n  a:\n    triggers: [x]\nrulez:\n  b:\n    triggers: [y]\n"
        with pytest.raises(ValueError, match="rulez"):
            _load(tmp_path, text)

    def test_non_string_rule_name_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="rule name"):
            _load(tmp_path, "rules:\n  42:\n    triggers: [x]\n")

    def test_shipped_sample_rules_still_load(self) -> None:
        assert len(parse_rules_yaml(SAMPLE_RULES_PATH)) == 4


class TestFileHandling:
    def test_syntax_error_is_a_value_error_naming_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "broken_rules.yaml"
        path.write_text("rules:\n  a: [unclosed\n", encoding="utf-8")
        with pytest.raises(ValueError, match="broken_rules.yaml"):
            parse_rules_yaml(path)

    def test_python_tags_are_refused_as_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "evil.yaml"
        path.write_text("rules: !!python/object/apply:os.getcwd []\n", encoding="utf-8")
        with pytest.raises(ValueError, match="evil.yaml"):
            parse_rules_yaml(path)

    def test_directory_is_a_value_error_naming_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=tmp_path.name):
            parse_rules_yaml(tmp_path)

    def test_undecodable_bytes_are_a_value_error_naming_the_file(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "latin1_rules.yaml"
        path.write_bytes(b"rules:\n  a:\n    triggers: [l\xf6schen]\n")
        with pytest.raises(ValueError, match="latin1_rules.yaml"):
            parse_rules_yaml(path)

    def test_error_messages_name_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "named_rules.yaml"
        path.write_text("rules:\n  a:\n    triggers: []\n", encoding="utf-8")
        with pytest.raises(ValueError, match="named_rules.yaml"):
            parse_rules_yaml(path)

    def test_utf8_regardless_of_locale(self, tmp_path: Path) -> None:
        """Non-ASCII triggers load even when the locale encoding is ASCII."""
        path = tmp_path / "unicode_rules.yaml"
        path.write_text(
            "rules:\n  loeschen:\n    triggers: ['l\u00f6schen', '\u5220\u9664']\n"
            "    forbidden_prosody: {emotion: [angry]}\n",
            encoding="utf-8",
        )
        script = (
            "import sys\n"
            "from intent_engine.constitutional.rules import parse_rules_yaml\n"
            "rule = parse_rules_yaml(sys.argv[1])[0]\n"
            "print(ascii(list(rule.triggers)))\n"
        )
        repo_root = str(Path(__file__).resolve().parents[2])
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([repo_root, os.environ.get("PYTHONPATH", "")]),
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONUTF8": "0",
            "PYTHONCOERCECLOCALE": "0",
            "PYTHONIOENCODING": "ascii",
        }
        result = subprocess.run(
            [sys.executable, "-X", "utf8=0", "-c", script, str(path)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "['l\\xf6schen', '\\u5220\\u9664']"


class TestImmutability:
    def test_rule_collections_are_tuples(self) -> None:
        rule = ConstitutionalRule(
            name="r",
            triggers=["delete", "remove"],
            required_prosody=ProsodyCondition(emotion=["sincere"]),
            forbidden_prosody=ProsodyCondition(emotion=["angry"]),
        )
        assert rule.triggers == ("delete", "remove")
        assert rule.required_prosody is not None
        assert rule.required_prosody.emotion == ("sincere",)
        assert rule.forbidden_prosody is not None
        assert rule.forbidden_prosody.emotion == ("angry",)

    def test_parsed_rules_cannot_be_mutated_through_the_filter(self) -> None:
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        rule = cf.rules[0]
        with pytest.raises(AttributeError):
            rule.triggers.clear()  # type: ignore[attr-defined]
        assert rule.forbidden_prosody is not None
        with pytest.raises(AttributeError):
            rule.forbidden_prosody.emotion.clear()  # type: ignore[attr-defined]
        assert cf.rules[0].triggers
        decision = cf.evaluate(
            "delete_files", [SpanFeatures(0, 500, "x")], emotion="sarcastic"
        )
        assert decision.allow is False

    def test_rule_does_not_alias_the_callers_lists(self) -> None:
        triggers = ["delete"]
        emotions = ["angry"]
        rule = ConstitutionalRule(
            name="r",
            triggers=triggers,
            forbidden_prosody=ProsodyCondition(emotion=emotions),
        )
        triggers.clear()
        emotions.clear()
        assert rule.triggers == ("delete",)
        assert rule.forbidden_prosody is not None
        assert rule.forbidden_prosody.emotion == ("angry",)

    def test_bare_string_triggers_are_rejected(self) -> None:
        """A str would be iterated character by character."""
        with pytest.raises(TypeError, match="triggers"):
            ConstitutionalRule(name="r", triggers="delete")  # type: ignore[arg-type]

    def test_bare_string_emotion_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="emotion"):
            ProsodyCondition(emotion="sincere")  # type: ignore[arg-type]


class TestDirectConstructionIsValidatedToo:
    def test_unknown_pitch_variance(self) -> None:
        with pytest.raises(ValueError, match="pitch_variance"):
            ProsodyCondition(pitch_variance="hihg")

    def test_inverted_speaking_rate(self) -> None:
        with pytest.raises(ValueError, match="speaking_rate"):
            ProsodyCondition(speaking_rate=(5.0, 2.0))

    def test_speaking_rate_list_is_stored_as_tuple(self) -> None:
        cond = ProsodyCondition(speaking_rate=[2, 5])  # type: ignore[arg-type]
        assert cond.speaking_rate == (2.0, 5.0)

    def test_empty_emotion_label(self) -> None:
        with pytest.raises(ValueError, match="emotion"):
            ProsodyCondition(emotion=[""])

    def test_unknown_verification_method(self) -> None:
        with pytest.raises(ValueError, match="method"):
            Verification(method="bogus")

    def test_negative_retries(self) -> None:
        with pytest.raises(ValueError, match="retries"):
            Verification(retries=-1)

    def test_forbidden_prosody_with_measured_condition(self) -> None:
        with pytest.raises(ValueError, match="only 'emotion'"):
            ConstitutionalRule(
                name="r",
                triggers=["x"],
                forbidden_prosody=ProsodyCondition(speaking_rate=(8.0, 20.0)),
            )

    def test_trigger_without_words(self) -> None:
        with pytest.raises(ValueError, match="triggers"):
            ConstitutionalRule(name="r", triggers=["delete", "___"])

    def test_rule_needs_a_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            ConstitutionalRule(name="")

    def test_conditions_must_be_prosody_conditions(self) -> None:
        with pytest.raises(TypeError, match="required_prosody"):
            ConstitutionalRule(
                name="r",
                triggers=["x"],
                required_prosody={"emotion": ["calm"]},  # type: ignore[arg-type]
            )


class TestEmotionCaseInsensitive:
    def test_upper_case_emotion_still_hits_forbidden_rule(self) -> None:
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        decision = cf.evaluate("send_money", [SpanFeatures(0, 500, "x")], emotion="ANGRY")
        assert decision.allow is False
        assert decision.requires_verification is False

    def test_padded_emotion(self) -> None:
        cond = ProsodyCondition(emotion=["angry"])
        passed, _ = check_forbidden_prosody(cond, [], emotion=" Angry ")
        assert passed is False

    def test_required_emotion_compares_case_insensitively(self) -> None:
        cond = ProsodyCondition(emotion=["sincere"])
        passed, _ = check_required_prosody(cond, [], emotion="SINCERE")
        assert passed is True

    def test_rule_labels_are_normalised(self, tmp_path: Path) -> None:
        rules = _load(
            tmp_path,
            _rule("forbidden_prosody: {emotion: [Angry, ' SAD ']}"),
        )
        assert rules[0].forbidden_prosody is not None
        assert rules[0].forbidden_prosody.emotion == ("angry", "sad")
        cf = ConstitutionalFilter(rules)
        decision = cf.evaluate("delete_files", [SpanFeatures(0, 500, "x")], emotion="angry")
        assert decision.allow is False


class TestYamlFeatures:
    def test_anchors_and_merge_keys_still_work(self, tmp_path: Path) -> None:
        text = (
            "rules:\n"
            "  a: &base\n    triggers: [delete]\n    forbidden_prosody: {emotion: [angry]}\n"
            "  b:\n    <<: *base\n    triggers: [remove]\n"
        )
        rules = _load(tmp_path, text)
        assert [r.name for r in rules] == ["a", "b"]
        assert rules[1].triggers == ("remove",)
        assert rules[1].forbidden_prosody == rules[0].forbidden_prosody

    def test_unhashable_mapping_key_is_a_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalid YAML"):
            _load(tmp_path, "rules:\n  ? [a, b]\n  : {triggers: [x]}\n")

    def test_verification_object_must_be_a_verification(self) -> None:
        with pytest.raises(TypeError, match="verification"):
            ConstitutionalRule(
                name="r",
                triggers=["x"],
                verification={"method": "two_factor"},  # type: ignore[arg-type]
            )
