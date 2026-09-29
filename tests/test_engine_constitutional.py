"""The engine hands the constitutional filter the emotion *and* its confidence.

Upstream reports "no emotion" as ("neutral", 0.0).  If the filter took that
as a positive "neutral" reading, a rule that requires a calm speaker would
approve every single-utterance request.  These tests go through the public
engine API with the real filter and the shipped sample rules.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from prosody_protocol import SpanFeatures

from intent_engine.models.decision import Decision
from intent_engine.models.result import Result
from tests.conftest import create_mocked_engine, make_iml_document, make_span_features

RULES = str(Path(__file__).parent / "constitutional" / "sample_rules.yaml")


def _result(emotion: str, confidence: float, features: list | None = None) -> Result:
    return Result(
        text="send the money",
        emotion=emotion,
        confidence=confidence,
        iml="<iml/>",
        iml_document=make_iml_document(),
        suggested_tone=emotion if confidence >= 0.5 else "neutral",
        prosody_features=[] if features is None else features,
    )


class TestEvaluateResult:
    def test_angry_speaker_is_denied(self) -> None:
        engine = create_mocked_engine(constitutional_rules=RULES)

        decision = engine.evaluate_result("send_money", _result("angry", 0.8))

        assert not decision.allow
        assert not decision.requires_verification

    def test_calm_speaker_is_allowed(self) -> None:
        engine = create_mocked_engine(constitutional_rules=RULES)

        assert engine.evaluate_result("send_money", _result("calm", 0.8)).allow

    def test_no_emotion_reported_needs_verification(self) -> None:
        # ("neutral", 0.0) is what a single utterance gets: no evidence.
        engine = create_mocked_engine(constitutional_rules=RULES)

        decision = engine.evaluate_result("send_money", _result("neutral", 0.0))

        assert not decision.allow
        assert decision.requires_verification
        assert decision.verification_method == "two_factor"

    def test_confidence_is_what_separates_abstention_from_a_neutral_reading(self) -> None:
        engine = create_mocked_engine(constitutional_rules=RULES)

        # Without the confidence, the same label is taken at face value.
        assert engine.evaluate_intent("send_money", [], emotion="neutral").allow
        assert not engine.evaluate_intent(
            "send_money", [], emotion="neutral", emotion_confidence=0.0
        ).allow

    def test_text_only_fallback_result_fails_closed(self) -> None:
        # The analysis-failure fallback has no features and reports no emotion:
        # a rule that needs measured speech must not pass on that.
        engine = create_mocked_engine(constitutional_rules=RULES)

        decision = engine.evaluate_result("delete_files", _result("calm", 0.8, features=[]))

        assert not decision.allow

    def test_measured_calm_speech_passes_the_rule_that_needs_it(self) -> None:
        engine = create_mocked_engine(constitutional_rules=RULES)
        # steady pitch (about one semitone of spread) at a measured pace
        features = [
            SpanFeatures(
                start_ms=0, end_ms=1000, text="ok", f0_mean=185.0,
                f0_range=(180.0, 190.0), speech_rate=3.5,
            )
        ]

        decision = engine.evaluate_result("delete_files", _result("calm", 0.8, features))

        assert decision.allow, decision.denial_reason


class TestForwarding:
    def test_evaluate_intent_forwards_the_confidence_arguments(self) -> None:
        engine = create_mocked_engine()
        spy = MagicMock()
        spy.evaluate.return_value = Decision(allow=True)
        engine._filter = spy
        features = [make_span_features()]

        engine.evaluate_intent(
            "send_money",
            features,
            emotion="calm",
            emotion_confidence=0.3,
            min_emotion_confidence=0.6,
        )

        spy.evaluate.assert_called_once_with(
            "send_money",
            features,
            emotion="calm",
            context=None,
            emotion_confidence=0.3,
            min_emotion_confidence=0.6,
        )

    def test_evaluate_result_passes_what_the_result_carries(self) -> None:
        engine = create_mocked_engine()
        spy = MagicMock()
        spy.evaluate.return_value = Decision(allow=True)
        engine._filter = spy
        result = _result("sad", 0.7, [make_span_features()])

        engine.evaluate_result("send_money", result, context={"user": "u1"})

        spy.evaluate.assert_called_once_with(
            "send_money",
            result.prosody_features,
            emotion="sad",
            context={"user": "u1"},
            emotion_confidence=0.7,
            min_emotion_confidence=0.5,
        )

    def test_no_filter_configured_allows(self) -> None:
        engine = create_mocked_engine()

        assert engine.evaluate_result("delete_files", _result("angry", 0.9)).allow


class TestRulesPath:
    @pytest.mark.parametrize("path", ["", "   "])
    def test_an_empty_path_does_not_silently_disable_the_filter(self, path: str) -> None:
        with pytest.raises(ValueError, match="constitutional_rules"):
            create_mocked_engine(constitutional_rules=path)

    def test_a_missing_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            create_mocked_engine(constitutional_rules=str(tmp_path / "nope.yaml"))

    def test_none_runs_without_a_filter(self) -> None:
        assert create_mocked_engine(constitutional_rules=None)._filter is None
