"""Tests for the LLM base interface and InterpretationResult."""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace

import pytest

from intent_engine.errors import LLMError
from intent_engine.llm.base import (
    CORE_EMOTIONS,
    InterpretationResult,
    LLMProvider,
    chat_completion_text,
    normalize_emotion,
    parse_interpretation,
)


class TestInterpretationResult:
    def test_construction(self) -> None:
        result = InterpretationResult(
            intent="request_help",
            response_text="How can I help you?",
            suggested_emotion="empathetic",
        )
        assert result.intent == "request_help"
        assert result.response_text == "How can I help you?"
        assert result.suggested_emotion == "empathetic"

    def test_frozen(self) -> None:
        result = InterpretationResult(
            intent="greet",
            response_text="Hello!",
            suggested_emotion="joyful",
        )
        with pytest.raises(AttributeError):
            result.intent = "changed"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = InterpretationResult("a", "b", "c")
        b = InterpretationResult("a", "b", "c")
        assert a == b

    def test_inequality(self) -> None:
        a = InterpretationResult("a", "b", "c")
        b = InterpretationResult("x", "b", "c")
        assert a != b


class TestLLMProviderInterface:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_subclass_must_implement_interpret(self) -> None:
        class IncompleteLLM(LLMProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteLLM()  # type: ignore[abstract]

    def test_concrete_subclass(self) -> None:
        class ConcreteLLM(LLMProvider):
            async def interpret(
                self, iml_input: str, context: str | None = None
            ) -> InterpretationResult:
                return InterpretationResult(
                    intent="test",
                    response_text="test response",
                    suggested_emotion="neutral",
                )

        llm = ConcreteLLM()
        assert isinstance(llm, LLMProvider)

        result = asyncio.run(
            llm.interpret("<utterance>hello</utterance>")
        )
        assert isinstance(result, InterpretationResult)
        assert result.intent == "test"


class TestParseInterpretation:
    GOOD = '{"intent": "greet", "response_text": "Hello!", "suggested_emotion": "joyful"}'

    def test_plain_json(self) -> None:
        assert parse_interpretation(self.GOOD, "Test") == InterpretationResult(
            "greet", "Hello!", "joyful"
        )

    def test_values_are_stripped(self) -> None:
        raw = '{"intent": " greet ", "response_text": "\\nHi!\\n", "suggested_emotion": " Joyful"}'
        assert parse_interpretation(raw, "Test") == InterpretationResult(
            "greet", "Hi!", "joyful"
        )

    def test_extra_keys_are_ignored(self) -> None:
        raw = self.GOOD[:-1] + ', "confidence": 0.9}'
        assert parse_interpretation(raw, "Test").intent == "greet"

    def test_object_inside_prose_and_fence(self) -> None:
        raw = f"Sure! Here it is:\n```json\n{self.GOOD}\n```\nAnything else?"
        assert parse_interpretation(raw, "Test").response_text == "Hello!"

    def test_braces_inside_a_string_value_do_not_confuse_the_search(self) -> None:
        raw = 'Result: {"intent": "a", "response_text": "use {x} here", "suggested_emotion": "sad"}'
        assert parse_interpretation(raw, "Test").response_text == "use {x} here"

    def test_error_names_the_source_and_quotes_the_start_of_the_reply(self) -> None:
        with pytest.raises(LLMError, match=r"Widget returned non-JSON response: oops"):
            parse_interpretation("oops", "Widget")

    def test_deeply_nested_reply_is_an_llm_error(self) -> None:
        with pytest.raises(LLMError):
            parse_interpretation('{"a":' * 5000, "Test")

    def test_degenerate_reply_is_rejected_quickly(self) -> None:
        # A model stuck repeating itself can produce very long runs of whitespace or braces;
        # an unbounded search for the object (or a backtracking pattern) would take minutes.
        started = time.perf_counter()
        for raw in ("```json" + " " * 20_000 + "x", "{" * 20_000, "{ " * 10_000):
            with pytest.raises(LLMError):
                parse_interpretation(raw, "Test")
        assert time.perf_counter() - started < 10.0

    def test_error_snippet_is_bounded(self) -> None:
        with pytest.raises(LLMError) as info:
            parse_interpretation("x" * 5000, "Test")
        assert len(str(info.value)) < 300

    def test_missing_field_is_named(self) -> None:
        with pytest.raises(LLMError, match="'suggested_emotion'"):
            parse_interpretation('{"intent": "a", "response_text": "b"}', "Test")


class TestNormalizeEmotion:
    @pytest.mark.parametrize("label", CORE_EMOTIONS)
    def test_core_labels_pass_through(self, label: str) -> None:
        assert normalize_emotion(label) == label

    @pytest.mark.parametrize("label", ["Sad", "SAD", " sad\n"])
    def test_case_and_whitespace_are_ignored(self, label: str) -> None:
        assert normalize_emotion(label) == "sad"

    @pytest.mark.parametrize("label", ["excited", "warm", "sad and tired", ""])
    def test_unknown_labels_become_neutral(self, label: str) -> None:
        assert normalize_emotion(label) == "neutral"

    def test_the_label_is_only_logged_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG, logger="intent_engine.llm.base"):
            normalize_emotion("excited")

        assert any(r.levelno == logging.WARNING for r in caplog.records)
        loud = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
        assert loud
        assert all("excited" not in message for message in loud)
        assert any("excited" in r.getMessage() for r in caplog.records)


class TestChatCompletionText:
    def test_sdk_object(self) -> None:
        message = SimpleNamespace(content="hi", refusal=None)
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        assert chat_completion_text(response, "Test") == "hi"

    def test_llama_dict(self) -> None:
        response = {"choices": [{"message": {"content": "hi"}}]}
        assert chat_completion_text(response, "Test") == "hi"

    @pytest.mark.parametrize(
        "response",
        [{}, {"choices": []}, SimpleNamespace(choices=None), SimpleNamespace(choices=[])],
    )
    def test_no_choices(self, response: object) -> None:
        with pytest.raises(LLMError, match="Test returned no choices"):
            chat_completion_text(response, "Test")

    @pytest.mark.parametrize("content", [None, "", "  "])
    def test_no_content(self, content: str | None) -> None:
        response = {"choices": [{"message": {"content": content}}]}
        with pytest.raises(LLMError, match="Test returned no text"):
            chat_completion_text(response, "Test")

    def test_refusal_is_quoted(self) -> None:
        response = {"choices": [{"message": {"content": None, "refusal": "No can do."}}]}
        with pytest.raises(LLMError, match="refused to answer: No can do."):
            chat_completion_text(response, "Test")
