"""Trigger matching: separators, case and whole-word semantics (audit #38, #78)."""

from __future__ import annotations

from pathlib import Path

import pytest
from prosody_protocol import SpanFeatures

from intent_engine.constitutional import ConstitutionalFilter
from intent_engine.constitutional.evaluator import match_triggers
from intent_engine.constitutional.rules import parse_rules_yaml


class TestNormalisation:
    @pytest.mark.parametrize(
        ("intent", "trigger"),
        [
            ("delete_all_files", "delete all"),
            ("Delete all files", "delete_all"),
            ("delete-all-files", "delete all"),
            ("deleteAllFiles", "delete all"),
            ("send_money", "send money"),
            ("SEND_MONEY", "send-money"),
            ("delete_all_files", "delete_all_files"),
            ("please.delete/file", "delete"),
            ("  delete   file ", "delete file"),
            ("löschen_alle", "LÖSCHEN"),
            # Composed and decomposed forms of the same letter are equal.
            ("löschen", "löschen"),
            ("删除_文件", "删除"),
        ],
    )
    def test_matches(self, intent: str, trigger: str) -> None:
        assert match_triggers(intent, [trigger])


class TestWholeWordSequences:
    @pytest.mark.parametrize(
        ("intent", "trigger"),
        [
            # A trigger must not match inside a longer word ...
            ("undelete_file", "delete"),
            ("remove_tag", "remove all"),
            ("greet_user", "delete"),
            ("action_10", "action_1"),
            # ... nor across a word boundary that is not there.
            ("delete_files", "ete fil"),
            # The intent being a fragment of the trigger is not a match.
            ("end_call", "send_money"),
            ("end", "send_money"),
        ],
    )
    def test_does_not_match(self, intent: str, trigger: str) -> None:
        assert not match_triggers(intent, [trigger])

    def test_words_must_be_adjacent_and_in_order(self) -> None:
        assert not match_triggers("delete_my_files", ["delete files"])
        assert not match_triggers("files_delete", ["delete files"])

    def test_trigger_may_be_a_sub_sequence_of_a_longer_intent(self) -> None:
        assert match_triggers("please_delete_all_files_now", ["delete all files"])

    def test_generic_intent_does_not_match_more_specific_trigger(self) -> None:
        """Only the trigger phrase appearing in the intent counts."""
        assert not match_triggers("delete", ["delete_all_files"])


class TestDegenerateTriggers:
    @pytest.mark.parametrize("trigger", ["", "   ", "___", "!!!", "-"])
    def test_empty_trigger_matches_nothing(self, trigger: str) -> None:
        assert not match_triggers("delete_files", [trigger])
        assert not match_triggers("", [trigger])

    def test_a_bare_string_is_one_trigger_not_a_list_of_letters(self) -> None:
        assert match_triggers("delete_file", "delete")
        assert not match_triggers("a_b_c", "delete")
        assert not match_triggers("greet_a_user", "delete")

    def test_empty_intent_matches_nothing(self) -> None:
        assert not match_triggers("", ["delete"])
        assert not match_triggers("___", ["delete"])


class TestDocumentedExample:
    """The flagship README/spec rule must fire for the documented intent."""

    RULES = """\
rules:
  destructive_file_operations:
    triggers:
      - "delete all"
      - "remove everything"
      - "wipe"
    required_prosody:
      emotion: [calm]
    forbidden_prosody:
      emotion: [sarcastic, frustrated]
    verification:
      method: explicit_confirmation
      retries: 2
"""

    def _filter(self, tmp_path: Path) -> ConstitutionalFilter:
        path = tmp_path / "rules.yaml"
        path.write_text(self.RULES, encoding="utf-8")
        return ConstitutionalFilter(parse_rules_yaml(path))

    @pytest.mark.parametrize(
        "intent", ["delete_all_files", "delete all files", "Delete-All-Files"]
    )
    def test_sarcastic_destructive_command_is_denied(
        self, tmp_path: Path, intent: str
    ) -> None:
        decision = self._filter(tmp_path).evaluate(
            intent, [SpanFeatures(0, 500, "x")], emotion="sarcastic"
        )
        assert decision.allow is False
        assert decision.requires_verification is False

    def test_unrelated_intent_still_allowed(self, tmp_path: Path) -> None:
        decision = self._filter(tmp_path).evaluate(
            "undelete_file", [SpanFeatures(0, 500, "x")], emotion="sarcastic"
        )
        assert decision.allow is True
