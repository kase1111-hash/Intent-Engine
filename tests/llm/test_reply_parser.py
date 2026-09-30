"""The reply parser accepts one unambiguous object, quickly, or raises ``LLMError``.

The intent in the reply feeds the constitutional gate, so a reply from which two
different objects could be read is refused rather than resolved by guessing, and
a truncated reply never yields an object from inside the part that was cut off.
"""

from __future__ import annotations

import json
import sys
import time

import pytest

from intent_engine.errors import LLMError
from intent_engine.llm.base import InterpretationResult, parse_interpretation

GOOD = {"intent": "small_talk", "response_text": "Nice day.", "suggested_emotion": "calm"}
GOOD_JSON = json.dumps(GOOD)
OTHER_JSON = json.dumps({**GOOD, "intent": "delete_all_files"})

# Comfortably above the milliseconds these take; below what a quadratic scan needs.
_FAST_S = 1.0


def _parse(raw: str) -> InterpretationResult:
    return parse_interpretation(raw, "Test")


class TestOneObject:
    @pytest.mark.parametrize(
        "raw",
        [
            GOOD_JSON,
            f"Here you go:\n{GOOD_JSON}",
            f"{GOOD_JSON}\nHope that helps!",
            f"```json\n{GOOD_JSON}\n```",
            f"Sure {{thing}} and {{ not json }}.\n{GOOD_JSON}",
            f"{GOOD_JSON}\nLet me know if you need more (e.g. {{curly}} things).",
        ],
    )
    def test_is_accepted(self, raw: str) -> None:
        assert _parse(raw).intent == "small_talk"

    def test_a_nested_object_value_is_part_of_the_one_object(self) -> None:
        raw = json.dumps({**GOOD, "meta": {"intent": "chat", "tags": [{"a": 1}]}})
        assert _parse(raw).intent == "small_talk"

    def test_a_long_valid_reply_is_parsed_in_full(self) -> None:
        reply = json.dumps({**GOOD, "response_text": "word " * 200_000})
        started = time.perf_counter()
        assert len(_parse(f"Answer:\n{reply}\nDone.").response_text) > 900_000
        assert time.perf_counter() - started < _FAST_S


class TestAmbiguousReplies:
    def test_the_first_of_two_objects_does_not_win(self) -> None:
        raw = f"You said {OTHER_JSON}. My answer: {GOOD_JSON}"
        with pytest.raises(LLMError, match="more than one JSON object"):
            _parse(raw)

    def test_two_identical_objects_are_refused_too(self) -> None:
        with pytest.raises(LLMError, match="more than one JSON object"):
            _parse(f"{GOOD_JSON}\n{GOOD_JSON}")

    def test_a_truncated_outer_object_does_not_yield_an_inner_one(self) -> None:
        raw = (
            '{"intent":"cancel","meta":{"intent":"chat","response_text":"hi",'
            '"suggested_emotion":"calm"},"response_text":"trunc'
        )
        with pytest.raises(LLMError, match="non-JSON"):
            _parse(raw)

    def test_a_malformed_object_is_not_skipped_for_a_later_one(self) -> None:
        with pytest.raises(LLMError, match="non-JSON"):
            _parse(f'Format: {{"intent": string}}\n{GOOD_JSON}')

    def test_a_fenced_truncation_is_refused(self) -> None:
        with pytest.raises(LLMError, match="non-JSON"):
            _parse(f"```json\n{GOOD_JSON[:-15]}")


DEGENERATE = {
    "open braces": "{" * 200_000,
    "spaced braces": "{ " * 100_000,
    "open braces with quotes": '{"' * 100_000,
    "nested keys": '{"a":' * 100_000,
    "nested arrays": "[" * 200_000,
    "many candidates": '{"a" ' * 100_000,
    "long whitespace": "{" + " " * 500_000,
    "endless prose": "no object here " * 100_000,
    "many complete objects": '{"a":1}' * 50_000,
    "fence and whitespace": "```json" + " " * 200_000 + "x",
    "unterminated string": '{"intent": "' + "a" * 500_000,
}


class TestDegenerateReplies:
    @pytest.mark.parametrize("raw", list(DEGENERATE.values()), ids=list(DEGENERATE))
    def test_are_refused_quickly(self, raw: str) -> None:
        started = time.perf_counter()
        with pytest.raises(LLMError):
            _parse(raw)
        assert time.perf_counter() - started < _FAST_S


@pytest.mark.skipif(
    not hasattr(sys, "get_int_max_str_digits") or sys.get_int_max_str_digits() == 0,
    reason="this interpreter has no limit on the digits of an integer",
)
class TestHugeIntegers:
    @staticmethod
    def _digits() -> str:
        return "9" * (sys.get_int_max_str_digits() + 1)

    @pytest.mark.parametrize(
        "shape",
        [
            '{{"intent": {digits}}}',
            "{digits}",
            'note: {{"intent": {digits}}}',
            '{{"intent": "a", "response_text": "b", "suggested_emotion": "calm", "x": {digits}}}',
            '{{"intent": {digits}',
        ],
    )
    def test_are_an_llm_error_not_a_value_error(self, shape: str) -> None:
        with pytest.raises(LLMError):
            _parse(shape.format(digits=self._digits()))
