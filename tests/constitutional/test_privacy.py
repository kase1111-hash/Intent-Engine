"""Emotional data stays out of logs and denial reasons (audit #46)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from prosody_protocol import SpanFeatures

from intent_engine.constitutional import ConstitutionalFilter
from intent_engine.constitutional.rules import (
    ConstitutionalRule,
    ProsodyCondition,
    Verification,
)

SAMPLE_RULES_PATH = Path(__file__).parent / "sample_rules.yaml"
FEATURES = [SpanFeatures(0, 500, "w", f0_range=(100.0, 110.0), speech_rate=3.0)]

# Distinctive labels: none of them may appear in rule names or messages by accident.
USER_EMOTIONS = ["sarcastic", "frustrated", "fearful", "joyful"]


def _filter() -> ConstitutionalFilter:
    return ConstitutionalFilter(
        [
            ConstitutionalRule(
                name="wire_guard",
                triggers=["wire"],
                required_prosody=ProsodyCondition(emotion=["calm", "sincere"]),
                forbidden_prosody=ProsodyCondition(emotion=["sarcastic", "fearful"]),
                verification=Verification(method="two_factor"),
            ),
            ConstitutionalRule(
                name="vault_guard",
                triggers=["vault"],
                required_prosody=ProsodyCondition(emotion=["calm"]),
            ),
        ]
    )


class TestLogsCarryNoEmotion:
    @pytest.mark.parametrize("intent", ["wire_funds", "open_vault"])
    @pytest.mark.parametrize("emotion", USER_EMOTIONS)
    def test_no_log_record_contains_the_emotion(
        self, caplog: pytest.LogCaptureFixture, intent: str, emotion: str
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="intent_engine"):
            _filter().evaluate(intent, FEATURES, emotion=emotion, emotion_confidence=0.9)
        assert caplog.records, "the evaluation should still be logged"
        for record in caplog.records:
            assert emotion not in record.getMessage().lower(), record.getMessage()

    @pytest.mark.parametrize("emotion", USER_EMOTIONS)
    def test_operators_still_learn_which_rule_denied(
        self, caplog: pytest.LogCaptureFixture, emotion: str
    ) -> None:
        with caplog.at_level(logging.INFO, logger="intent_engine"):
            _filter().evaluate("open_vault", FEATURES, emotion=emotion, emotion_confidence=0.9)
        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
        assert any("vault_guard" in m and "denied" in m for m in messages)

    def test_verification_is_logged_with_rule_and_method(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="intent_engine"):
            _filter().evaluate("wire_funds", FEATURES, emotion="joyful", emotion_confidence=0.9)
        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
        assert any(
            "wire_guard" in m and "verification" in m and "two_factor" in m for m in messages
        )


class TestDenialReasonCarriesNoEmotion:
    @pytest.mark.parametrize("emotion", USER_EMOTIONS)
    def test_reasons_do_not_repeat_the_emotion_or_reveal_the_policy(self, emotion: str) -> None:
        cf = _filter()
        for intent in ("wire_funds", "open_vault"):
            decision = cf.evaluate(intent, FEATURES, emotion=emotion, emotion_confidence=0.9)
            assert decision.allow is False
            reason = (decision.denial_reason or "").lower()
            assert emotion not in reason
            # The rule's own lists are configuration, not something to hand to the caller.
            assert "calm" not in reason
            assert "sincere" not in reason

    def test_forbidden_reason_stays_useful(self) -> None:
        decision = _filter().evaluate("wire_funds", FEATURES, emotion="sarcastic")
        assert decision.denial_reason == "Rule 'wire_guard': Forbidden emotion detected"

    def test_required_reason_says_whether_the_emotion_was_unknown(self) -> None:
        cf = _filter()
        unknown = cf.evaluate("open_vault", FEATURES, emotion="neutral", emotion_confidence=0.0)
        assert unknown.denial_reason is not None
        assert "Required emotion" in unknown.denial_reason
        assert "unknown" in unknown.denial_reason
        mismatch = cf.evaluate("open_vault", FEATURES, emotion="joyful", emotion_confidence=0.9)
        assert mismatch.denial_reason is not None
        assert "Required emotion" in mismatch.denial_reason
        assert "unknown" not in mismatch.denial_reason

    def test_measurement_reasons_keep_their_detail(self) -> None:
        cf = ConstitutionalFilter.from_yaml(SAMPLE_RULES_PATH)
        decision = cf.evaluate(
            "emergency", [SpanFeatures(0, 500, "w", speech_rate=2.0)], emotion=None
        )
        assert decision.denial_reason is not None
        assert "speaking_rate" in decision.denial_reason
        assert "2.0" in decision.denial_reason
