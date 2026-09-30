"""The gate weighs every believed emotion of a turn, not only ``Result.emotion``.

``Result.emotion`` is the most confident utterance of the recording, which is
the right summary for a display or a tone hint but not for a safety gate: a
forbidden emotion in a less confident utterance of the same turn must still
count.  ``IntentEngine.evaluate_result`` therefore evaluates the primary
emotion and every other distinct, believed ``(emotion, confidence)`` pair of
``Result.iml_document`` and returns the most restrictive decision.  An
utterance the assembler abstained on is no evidence either way (treating it as
"unknown" would fail every required emotion list on every multi-sentence turn).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from prosody_protocol import IMLDocument, ProsodyMapping, ProsodyProfile, Utterance

from intent_engine.constitutional.filter import ConstitutionalFilter
from intent_engine.constitutional.rules import (
    ConstitutionalRule,
    ProsodyCondition,
    Verification,
)
from intent_engine.engine import IntentEngine
from intent_engine.models.decision import Decision
from intent_engine.models.result import Result
from tests.conftest import create_mocked_engine

Pair = tuple[str | None, float | None]

#: A forbidden emotion and nothing else: an unknown emotion is not blocked.
FORBID_ANGRY = ConstitutionalRule(
    name="no_anger",
    triggers=["delete"],
    forbidden_prosody=ProsodyCondition(emotion=["angry"]),
)
#: The README's shape: calm required (verification otherwise), angry blocked.
FLAGSHIP = ConstitutionalRule(
    name="needs_calm",
    triggers=["delete"],
    required_prosody=ProsodyCondition(emotion=["calm"]),
    forbidden_prosody=ProsodyCondition(emotion=["angry", "frustrated", "sarcastic"]),
    verification=Verification("explicit_confirmation"),
)


def _engine(rule: ConstitutionalRule) -> IntentEngine:
    engine = create_mocked_engine()
    engine._filter = ConstitutionalFilter([rule])
    return engine


def _turn(*pairs: Pair, primary: Pair | None = None) -> Result:
    """A Result whose document carries one utterance per pair."""
    doc = IMLDocument(
        utterances=tuple(Utterance(emotion=e, confidence=c) for e, c in pairs),
        version="0.1.0",
    )
    emotion, confidence = primary or IntentEngine._document_emotion(doc)
    return Result(
        text="delete all my files",
        emotion=emotion,
        confidence=confidence,
        iml="<iml/>",
        iml_document=doc,
        suggested_tone=emotion if confidence >= 0.5 else "neutral",
        prosody_features=[],
    )


def _hard_deny(decision: Decision) -> bool:
    return not decision.allow and not decision.requires_verification


class TestHandBuiltTurns:
    """No audio: the Result is built from the pairs the assembler could produce."""

    @pytest.mark.parametrize(
        "pairs",
        [
            (("calm", 0.9), ("angry", 0.6)),
            (("angry", 0.6), ("calm", 0.9)),
            ((None, None), ("calm", 0.9), (None, None), ("angry", 0.54)),
        ],
    )
    def test_forbidden_emotion_in_a_less_confident_utterance_is_denied(
        self, pairs: tuple[Pair, ...]
    ) -> None:
        turn = _turn(*pairs)
        assert turn.emotion == "calm"  # the primary emotion is unchanged

        for rule in (FORBID_ANGRY, FLAGSHIP):
            decision = _engine(rule).evaluate_result("delete_all_files", turn)
            assert _hard_deny(decision), decision
            assert "Forbidden emotion detected" in (decision.denial_reason or "")

    def test_denial_reason_still_never_repeats_the_label(self) -> None:
        turn = _turn(("calm", 0.9), ("angry", 0.6))

        decision = _engine(FLAGSHIP).evaluate_result("delete_all_files", turn)

        assert "angry" not in (decision.denial_reason or "")

    def test_believed_emotion_outside_the_required_list_needs_verification(self) -> None:
        turn = _turn(("calm", 0.9), ("sad", 0.5))

        decision = _engine(FLAGSHIP).evaluate_result("delete_all_files", turn)

        assert not decision.allow
        assert decision.requires_verification
        assert decision.verification_method == "explicit_confirmation"

    def test_a_hard_deny_beats_a_verification_from_another_utterance(self) -> None:
        turn = _turn(("sad", 0.7), ("calm", 0.6), ("angry", 0.55))

        assert _hard_deny(_engine(FLAGSHIP).evaluate_result("delete_all_files", turn))

    def test_abstaining_utterances_do_not_count_as_unknown_emotion(self) -> None:
        # Calibration sentences always abstain: they must not turn a calm
        # turn into "emotion unknown" (which would make allow unreachable).
        turn = _turn((None, None), (None, None), (None, None), ("calm", 0.9))

        assert _engine(FLAGSHIP).evaluate_result("delete_all_files", turn).allow

    def test_every_utterance_calm_is_allowed(self) -> None:
        turn = _turn(("calm", 0.9), ("calm", 0.7), ("calm", 0.6))

        assert _engine(FLAGSHIP).evaluate_result("delete_all_files", turn).allow

    def test_an_emotion_below_the_confidence_threshold_is_not_believed(self) -> None:
        # Below 0.5 the filter treats the label as unknown, so it is neither
        # forbidden nor a reason to fail the required list.
        turn = _turn(("calm", 0.9), ("angry", 0.3))

        assert _engine(FLAGSHIP).evaluate_result("delete_all_files", turn).allow

    def test_no_emotion_anywhere_still_needs_verification(self) -> None:
        turn = _turn((None, None), (None, None))
        assert (turn.emotion, turn.confidence) == ("neutral", 0.0)

        decision = _engine(FLAGSHIP).evaluate_result("delete_all_files", turn)

        assert not decision.allow
        assert decision.requires_verification
        assert "emotion unknown" in (decision.denial_reason or "")

    def test_forbidden_only_rule_allows_a_turn_without_forbidden_emotion(self) -> None:
        turn = _turn(("calm", 0.9), ("sad", 0.7), (None, None))

        assert _engine(FORBID_ANGRY).evaluate_result("delete_all_files", turn).allow

    def test_the_result_own_values_still_count_when_the_document_disagrees(self) -> None:
        # A caller-supplied emotion (no emotion in the document) is honoured.
        turn = _turn((None, None), primary=("angry", 0.8))

        assert _hard_deny(_engine(FORBID_ANGRY).evaluate_result("delete_all_files", turn))

    def test_a_document_without_utterances_falls_back_to_the_result_values(self) -> None:
        empty = Result(
            text="",
            emotion="angry",
            confidence=0.8,
            iml="<iml/>",
            iml_document=IMLDocument(utterances=(), version="0.1.0"),
            suggested_tone="angry",
            prosody_features=[],
        )

        assert _hard_deny(_engine(FORBID_ANGRY).evaluate_result("delete_all_files", empty))

    @pytest.mark.parametrize("document", [None, object()])
    def test_a_missing_document_falls_back_to_the_result_values(self, document: Any) -> None:
        calm = Result(
            text="",
            emotion="calm",
            confidence=0.9,
            iml="<iml/>",
            iml_document=document,
            suggested_tone="calm",
            prosody_features=[],
        )
        angry = Result(
            text="",
            emotion="angry",
            confidence=0.9,
            iml="<iml/>",
            iml_document=document,
            suggested_tone="angry",
            prosody_features=[],
        )

        assert _engine(FLAGSHIP).evaluate_result("delete_all_files", calm).allow
        assert _hard_deny(_engine(FLAGSHIP).evaluate_result("delete_all_files", angry))

    def test_each_distinct_pair_is_evaluated_once(self) -> None:
        engine = create_mocked_engine()
        spy = MagicMock()
        spy.evaluate.return_value = Decision(allow=True)
        engine._filter = spy
        turn = _turn(("calm", 0.9), ("calm", 0.9), (None, None), ("angry", 0.6), ("angry", 0.6))

        engine.evaluate_result("delete_all_files", turn)

        emotions = [
            (call.kwargs["emotion"], call.kwargs["emotion_confidence"])
            for call in spy.evaluate.call_args_list
        ]
        assert emotions == [("calm", 0.9), ("angry", 0.6)]

    def test_without_a_filter_everything_is_allowed(self) -> None:
        engine = create_mocked_engine()

        assert engine.evaluate_result("delete_all_files", _turn(("angry", 0.9))).allow


# -- recordings through the real analyzer, assembler and filter (only STT is faked) --


@pytest.fixture()
def audio() -> Any:
    """The synthetic-audio helpers (skips when numpy/parselmouth are missing)."""
    pytest.importorskip("numpy")
    pytest.importorskip("parselmouth")
    from tests import synth_audio

    return synth_audio


#: A sad delivery deep enough for the classifier to be surer of it (0.6) than
#: of the angry sentence (0.54) that comes next to it.
DEEP_SAD = {"f0": 105, "gain_db": -28, "syl": 2.2}

_CALM_ON_NORMAL_PITCH = ProsodyProfile(
    profile_version="1.0.0",
    user_id="user-1",
    description=None,
    mappings=(ProsodyMapping({"pitch": "normal"}, "calm", 0.7),),
)


def _process(engine: IntentEngine, wav: Path) -> Result:
    return asyncio.run(engine.process_voice_input(str(wav), use_cache=False))


def _emotions(result: Result) -> list[Pair]:
    return [(u.emotion, u.confidence) for u in result.iml_document.utterances]


class TestRecordings:
    @pytest.mark.parametrize("rule", [FORBID_ANGRY, FLAGSHIP], ids=["forbid", "flagship"])
    @pytest.mark.parametrize("order", ["sad_then_angry", "angry_then_sad"])
    def test_angry_next_to_a_more_confident_sad_sentence_is_denied(
        self, audio: Any, tmp_path: Path, rule: ConstitutionalRule, order: str
    ) -> None:
        turn = [DEEP_SAD, audio.ANGRY] if order == "sad_then_angry" else [audio.ANGRY, DEEP_SAD]
        wav = tmp_path / "turn.wav"
        engine = audio.engine_for(audio.write_recording(wav, [audio.NEUTRAL] * 3 + turn))
        engine._filter = ConstitutionalFilter([rule])

        result = _process(engine, wav)

        believed = {e for e, c in _emotions(result) if e is not None}
        assert believed == {"sad", "angry"}  # the recording says what the test needs
        assert (result.emotion, result.confidence) == ("sad", 0.6)  # unchanged summary
        decision = engine.evaluate_result("delete_all_files", result)
        assert _hard_deny(decision), decision

    def test_angry_after_a_more_confident_joyful_sentence_is_denied(
        self, audio: Any, tmp_path: Path
    ) -> None:
        wav = tmp_path / "turn.wav"
        aligns = audio.write_recording(wav, [audio.NEUTRAL] * 4 + [audio.EXCITED, audio.ANGRY])
        engine = audio.engine_for(aligns)
        engine._filter = ConstitutionalFilter([FLAGSHIP])

        result = _process(engine, wav)

        assert result.emotion == "joyful"
        decision = engine.evaluate_result("delete_all_files", result)
        assert _hard_deny(decision), decision

    def test_angry_after_calm_sentences_from_a_profile_is_denied(
        self, audio: Any, tmp_path: Path
    ) -> None:
        wav = tmp_path / "turn.wav"
        aligns = audio.write_recording(wav, [audio.NEUTRAL] * 3 + [audio.ANGRY])
        engine = audio.engine_for(aligns)
        engine.set_profile(_CALM_ON_NORMAL_PITCH)
        engine._filter = ConstitutionalFilter([FLAGSHIP])

        result = _process(engine, wav)

        assert (result.emotion, result.confidence) == ("calm", 1.0)
        assert ("angry", pytest.approx(0.54, abs=0.05)) == _emotions(result)[-1]
        decision = engine.evaluate_result("delete_all_files", result)
        assert _hard_deny(decision), decision

    def test_a_turn_that_is_calm_throughout_is_still_allowed(
        self, audio: Any, tmp_path: Path
    ) -> None:
        wav = tmp_path / "turn.wav"
        aligns = audio.write_recording(wav, [audio.NEUTRAL] * 4)
        engine = audio.engine_for(aligns)
        engine.set_profile(_CALM_ON_NORMAL_PITCH)
        engine._filter = ConstitutionalFilter([FLAGSHIP])

        result = _process(engine, wav)

        assert {e for e, _ in _emotions(result)} == {"calm"}
        assert engine.evaluate_result("delete_all_files", result).allow

    def test_a_single_angry_sentence_is_still_denied(self, audio: Any, tmp_path: Path) -> None:
        wav = tmp_path / "turn.wav"
        aligns = audio.write_recording(wav, [audio.NEUTRAL] * 3 + [audio.ANGRY])
        engine = audio.engine_for(aligns)
        engine._filter = ConstitutionalFilter([FLAGSHIP])

        result = _process(engine, wav)

        assert result.emotion == "angry"
        assert _hard_deny(engine.evaluate_result("delete_all_files", result))
