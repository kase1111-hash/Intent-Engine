"""Tests for the prosody-aware system prompts."""

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

    def test_mentions_iml_tags(self) -> None:
        assert "<utterance>" in SYSTEM_PROMPT
        assert "<prosody>" in SYSTEM_PROMPT
        assert "<emphasis>" in SYSTEM_PROMPT
        assert "<pause>" in SYSTEM_PROMPT
        assert "<segment>" in SYSTEM_PROMPT

    def test_mentions_core_emotions(self) -> None:
        core_emotions = [
            "neutral", "sincere", "sarcastic", "frustrated", "joyful",
            "uncertain", "angry", "sad", "fearful", "surprised",
            "disgusted", "calm", "empathetic",
        ]
        for emotion in core_emotions:
            assert emotion in SYSTEM_PROMPT, f"Missing core emotion: {emotion}"

    def test_mentions_prosody_attributes(self) -> None:
        assert "pitch" in SYSTEM_PROMPT
        assert "pitch_contour" in SYSTEM_PROMPT
        assert "volume" in SYSTEM_PROMPT
        assert "rate" in SYSTEM_PROMPT
        assert "quality" in SYSTEM_PROMPT

    def test_mentions_pitch_contour_patterns(self) -> None:
        patterns = ["rise", "fall", "fall-rise", "rise-fall", "flat"]
        for pattern in patterns:
            assert pattern in SYSTEM_PROMPT, f"Missing contour pattern: {pattern}"

    def test_requests_json_response(self) -> None:
        assert "JSON" in SYSTEM_PROMPT
        assert "intent" in SYSTEM_PROMPT
        assert "response_text" in SYSTEM_PROMPT
        assert "suggested_emotion" in SYSTEM_PROMPT

    def test_emphasis_levels(self) -> None:
        assert "strong" in SYSTEM_PROMPT
        assert "moderate" in SYSTEM_PROMPT
        assert "reduced" in SYSTEM_PROMPT


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
