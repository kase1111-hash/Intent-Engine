"""Tests for the prosody-aware system prompts.

What the prompt teaches is checked against the IML validator, the spec's
vocabulary and the assembler's real output in ``test_prompt_iml_conformance.py``.
"""

from __future__ import annotations

from intent_engine.llm.prompts import JSON_RESPONSE_SCHEMA, PROMPT_VERSION, SYSTEM_PROMPT


class TestPromptVersion:
    def test_version_is_string(self) -> None:
        assert isinstance(PROMPT_VERSION, str)

    def test_version_format(self) -> None:
        parts = PROMPT_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


class TestSystemPrompt:
    def test_is_non_empty_string(self) -> None:
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100


class TestJsonResponseSchema:
    def test_schema_structure(self) -> None:
        assert JSON_RESPONSE_SCHEMA["type"] == "object"
        assert "properties" in JSON_RESPONSE_SCHEMA
        assert "required" in JSON_RESPONSE_SCHEMA

    def test_required_fields(self) -> None:
        required = JSON_RESPONSE_SCHEMA["required"]
        assert "intent" in required
        assert "response_text" in required
        assert "suggested_emotion" in required

    def test_no_additional_properties(self) -> None:
        assert JSON_RESPONSE_SCHEMA["additionalProperties"] is False
