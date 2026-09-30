"""Trigger matching survives invisible characters and acronym runs.

The intent label is written by an LLM, so it is not always tidy snake_case.  A
format character inside a word (soft hyphen, zero-width joiner, NUL) used to
split the word (``del ete``) and a run of capitals (``deleteALLFiles``) was
never split, so guarded intents slipped past their triggers.  The intent is
read both ways a reader could mean an ignorable character (as a separator or
as nothing) and matches when any reading matches, so the result is fail-closed
whichever was meant.
"""

from __future__ import annotations

import pytest

from intent_engine.constitutional.evaluator import match_triggers
from intent_engine.constitutional.rules import normalize_phrase, phrase_readings


class TestInvisibleCharacters:
    @pytest.mark.parametrize(
        "intent",
        [
            "del​ete_all_files",  # zero-width space inside a word
            "del­ete_all_files",  # soft hyphen
            "del‍ete_all_files",  # zero-width joiner
            "del‌ete_all_files",  # zero-width non-joiner
            "del⁠ete_all_files",  # word joiner
            "del﻿ete_all_files",  # byte order mark
            "del\x00ete_all_files",  # NUL
            "de̸lete_all_files",  # combining mark that does not compose
            "delete​all_files",  # ... and as a separator, as before
            "delete⁠all",
            "delete﻿all",
            "delete​​all",
            "delete_​all_files",
            "del​​ete_all",
        ],
    )
    def test_delete_all_is_still_recognised(self, intent: str) -> None:
        assert match_triggers(intent, ["delete all"]), repr(intent)

    @pytest.mark.parametrize(
        "intent",
        ["undelete_all", "delete_my_files", "un​_delete_files", "del​ete_files_all"],
    )
    def test_whole_word_matching_is_kept(self, intent: str) -> None:
        assert not match_triggers(intent, ["delete all"]), repr(intent)

    def test_words_are_not_glued_together_when_a_separator_was_meant(self) -> None:
        # "send" + ZWSP + "money" must still read as two words
        assert match_triggers("send​money_now", ["send money"])
        assert match_triggers("send​money_now", ["send"])
        assert match_triggers("send​money_now", ["money now"])

    def test_normalize_phrase_itself_is_unchanged(self) -> None:
        assert normalize_phrase("del​ete_all_files") == "del ete all files"
        assert normalize_phrase("Delete-ALL_files") == "delete all files"


class TestAcronymRuns:
    @pytest.mark.parametrize(
        "intent",
        ["deleteALLFiles", "deleteAllFiles", "DeleteALLFiles", "delete_ALLFiles", "deleteAll"],
    )
    def test_camel_case_with_a_capital_run(self, intent: str) -> None:
        assert match_triggers(intent, ["delete all"]), intent

    def test_the_camel_case_reading_without_acronym_splitting_still_applies(self) -> None:
        # "IDs" reads "ids" (not "i ds"), so a trigger written that way still fires
        assert match_triggers("deleteUserIDs", ["user ids"])
        assert match_triggers("getIDs", ["get ids"])
        assert match_triggers("parseHTTPResponse", ["http response"])

    def test_whole_words_only(self) -> None:
        assert not match_triggers("deleteALLFiles", ["delete al"])
        assert not match_triggers("HTTPServer", ["tp serv"])


class TestReadings:
    def test_a_plain_label_has_one_reading(self) -> None:
        assert phrase_readings("delete_all_files") == ("delete all files",)

    def test_readings_are_distinct_and_ordered(self) -> None:
        readings = phrase_readings("del​ete_ALLFiles")

        assert readings[0] == normalize_phrase("del​ete_ALLFiles")
        assert len(set(readings)) == len(readings)
        assert "delete all files" in readings

    def test_empty_and_blank_labels(self) -> None:
        assert phrase_readings("") == ("",)
        assert not match_triggers("", ["delete"])
        assert not match_triggers("​​", ["delete"])
