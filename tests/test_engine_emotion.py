"""Result.emotion / confidence / suggested_tone come from the assembled IML.

The assembler measures each utterance against the speaker's baseline and
abstains when it cannot tell, so ``Result`` must report exactly what the IML
handed to the LLM says.  The audio tests run the real ``ProsodyAnalyzer``,
``IMLAssembler`` and ``IMLValidator``; only STT is faked.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from prosody_protocol import IMLDocument, Utterance

from intent_engine.constitutional.filter import ConstitutionalFilter
from intent_engine.constitutional.rules import ConstitutionalRule, ProsodyCondition
from intent_engine.engine import IntentEngine
from tests.conftest import assert_valid_iml, create_mocked_engine, make_iml_document

pytest.importorskip("numpy")
pytest.importorskip("parselmouth")

from tests.synth_audio import (  # noqa: E402
    ANGRY,
    EXCITED,
    NEUTRAL,
    SAD,
    engine_for,
    write_recording,
)


def _iml_emotions(result) -> list[tuple[str | None, float | None]]:
    return [(u.emotion, u.confidence) for u in result.iml_document.utterances]


class TestResultMatchesIML:
    @pytest.mark.parametrize(
        ("label", "last"),
        [("joyful", EXCITED), ("angry", ANGRY), ("sad", SAD)],
    )
    def test_emotion_reported_in_iml_is_reported_in_result(
        self, tmp_path: Path, label: str, last: dict[str, float]
    ) -> None:
        wav = tmp_path / "turn.wav"
        aligns = write_recording(wav, [NEUTRAL] * 4 + [last])
        engine = engine_for(aligns)

        result = asyncio.run(engine.process_voice_input(str(wav), use_cache=False))

        in_iml = [(e, c) for e, c in _iml_emotions(result) if e is not None]
        assert in_iml, "synthetic audio should produce an emotion in the IML"
        assert (result.emotion, result.confidence) == in_iml[-1]
        assert result.emotion == label
        assert result.confidence >= 0.5
        assert result.suggested_tone == result.emotion
        assert_valid_iml(result.iml)

    def test_single_utterance_abstains(self, tmp_path: Path) -> None:
        wav = tmp_path / "one.wav"
        aligns = write_recording(wav, [EXCITED])
        engine = engine_for(aligns)

        result = asyncio.run(engine.process_voice_input(str(wav), use_cache=False))

        assert all(u.emotion is None for u in result.iml_document.utterances)
        assert (result.emotion, result.confidence) == ("neutral", 0.0)
        assert result.suggested_tone == "neutral"

    def test_engine_and_assembler_share_one_classifier(self) -> None:
        with patch("intent_engine.engine.IMLAssembler") as assembler_cls:
            engine = create_mocked_engine()
        assert assembler_cls.call_args.kwargs["emotion_classifier"] is engine._emotion_classifier

    def test_forbidden_emotion_rule_sees_the_detected_emotion(self, tmp_path: Path) -> None:
        wav = tmp_path / "angry.wav"
        aligns = write_recording(wav, [NEUTRAL] * 4 + [ANGRY])
        engine = engine_for(aligns)
        engine._filter = ConstitutionalFilter(
            [
                ConstitutionalRule(
                    name="delete_files",
                    triggers=["delete"],
                    forbidden_prosody=ProsodyCondition(emotion=["angry"]),
                )
            ]
        )

        result = asyncio.run(engine.process_voice_input(str(wav), use_cache=False))
        decision = engine.evaluate_intent(
            "delete_files", result.prosody_features, emotion=result.emotion
        )

        assert decision.allow is False


def _doc(*emotions: tuple[str | None, float | None]) -> IMLDocument:
    return IMLDocument(
        utterances=tuple(Utterance(emotion=e, confidence=c) for e, c in emotions),
        version="0.1.0",
    )


class TestDocumentEmotion:
    def test_no_emotion_is_neutral_zero(self) -> None:
        assert IntentEngine._document_emotion(_doc((None, None), (None, None))) == ("neutral", 0.0)

    def test_empty_document_is_neutral_zero(self) -> None:
        assert IntentEngine._document_emotion(make_iml_document()) == ("neutral", 0.0)

    def test_picks_most_confident_utterance(self) -> None:
        doc = _doc(("angry", 0.54), (None, None), ("joyful", 0.72), ("sad", 0.6))
        assert IntentEngine._document_emotion(doc) == ("joyful", 0.72)

    def test_tie_goes_to_the_later_utterance(self) -> None:
        doc = _doc(("angry", 0.6), ("sad", 0.6))
        assert IntentEngine._document_emotion(doc) == ("sad", 0.6)

    def test_emotion_without_confidence_is_ignored(self) -> None:
        assert IntentEngine._document_emotion(_doc(("angry", None))) == ("neutral", 0.0)
