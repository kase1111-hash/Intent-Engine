"""Tests for the LLM base interface and InterpretationResult."""

from __future__ import annotations

import asyncio

import pytest

from intent_engine.llm.base import InterpretationResult, LLMProvider


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

        result = asyncio.get_event_loop().run_until_complete(
            llm.interpret("<utterance>hello</utterance>")
        )
        assert isinstance(result, InterpretationResult)
        assert result.intent == "test"
