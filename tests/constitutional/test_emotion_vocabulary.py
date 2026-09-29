"""Rules that use labels the built-in classifier never emits say so (audit #43)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from intent_engine.constitutional.rules import (
    CLASSIFIER_EMOTIONS,
    CORE_EMOTIONS,
    parse_rules_yaml,
)


class TestClassifierVocabulary:
    def test_the_classifier_labels_are_core_labels(self) -> None:
        assert CLASSIFIER_EMOTIONS < CORE_EMOTIONS

    def test_matches_what_the_installed_classifier_can_emit(self) -> None:
        module = pytest.importorskip("prosody_protocol.emotion_classifier")
        labels = getattr(module, "_LABELS", None)
        if labels is None:
            pytest.skip("prosody_protocol no longer lists its labels in _LABELS")
        assert set(labels) == CLASSIFIER_EMOTIONS, (
            "CLASSIFIER_EMOTIONS is out of date with the labels "
            "prosody_protocol's RuleBasedEmotionClassifier can emit"
        )

    def _load(self, tmp_path: Path, emotions: str) -> None:
        path = tmp_path / "rules.yaml"
        path.write_text(
            f"rules:\n  vault_guard:\n    triggers: [vault]\n"
            f"    required_prosody: {{emotion: {emotions}}}\n",
            encoding="utf-8",
        )
        parse_rules_yaml(path)

    def test_warns_about_labels_the_classifier_never_emits(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="intent_engine"):
            self._load(tmp_path, "[calm, sincere, sarcastic]")
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "vault_guard" in warnings[0]
        assert "sincere" in warnings[0] and "sarcastic" in warnings[0]
        assert "calm" not in warnings[0]
        assert "RuleBasedEmotionClassifier" in warnings[0]

    def test_no_warning_when_every_label_is_producible(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="intent_engine"):
            self._load(tmp_path, "[neutral, calm, sad, angry, joyful, fearful]")
        assert not [r for r in caplog.records if r.levelno >= logging.INFO]

    def test_labels_outside_the_core_vocabulary_are_still_reported(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="intent_engine"):
            self._load(tmp_path, "[calm, confident]")
        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("confident" in m and "core vocabulary" in m for m in messages)

    def test_case_does_not_hide_a_label(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="intent_engine"):
            self._load(tmp_path, "[Calm, SINCERE]")
        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(messages) == 1
        assert "sincere" in messages[0]
