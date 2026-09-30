"""The TTS adapters must not log the emotion they are asked to speak with.

The emotion comes from the LLM, or from the caller's ``suggested_tone`` (the
user's detected tone), and emotional data is treated as sensitive.  Nothing at
INFO or above may carry the label, or a setting that maps one-to-one onto it
such as eSpeak's rate and volume.  The label is logged at DEBUG only.
"""

from __future__ import annotations

import logging
import re
import sys
import types
from collections.abc import Callable

import pytest

from intent_engine.tts.base import EMOTION_VOICE_MAP, TTSProvider, normalize_emotion
from intent_engine.tts.coqui import CoquiTTS
from intent_engine.tts.elevenlabs import ElevenLabsTTS
from intent_engine.tts.espeak import ESpeakTTS
from tests.tts.fake_pyttsx3 import install_fake_pyttsx3
from tests.tts.test_coqui import StubModel

# Every core label, an unknown one, and one that needs normalising.
LABELS = [*EMOTION_VOICE_MAP, "overwhelmed", "Excited "]
Factory = Callable[[pytest.MonkeyPatch], TTSProvider]


def _elevenlabs(monkeypatch: pytest.MonkeyPatch) -> TTSProvider:
    client = types.SimpleNamespace(
        text_to_speech=types.SimpleNamespace(convert=lambda **kwargs: iter([b"audio"]))
    )
    module = types.ModuleType("elevenlabs")
    module.ElevenLabs = lambda **kwargs: client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "elevenlabs", module)
    return ElevenLabsTTS(api_key="k")


def _coqui(monkeypatch: pytest.MonkeyPatch) -> TTSProvider:
    tts = CoquiTTS()
    tts._tts = StubModel()
    return tts


def _espeak(monkeypatch: pytest.MonkeyPatch) -> TTSProvider:
    install_fake_pyttsx3(monkeypatch)
    return ESpeakTTS()


ADAPTERS: dict[str, Factory] = {"elevenlabs": _elevenlabs, "coqui": _coqui, "espeak": _espeak}


def _words(label: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(label.strip().lower())}\b", re.IGNORECASE)


@pytest.mark.parametrize("adapter", list(ADAPTERS))
@pytest.mark.parametrize("label", LABELS)
async def test_the_emotion_is_not_logged_at_info_or_above(
    adapter: str,
    label: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tts = ADAPTERS[adapter](monkeypatch)

    with caplog.at_level(logging.INFO):
        result = await tts.synthesize("Hello there", emotion=label)

    assert result.audio_data
    assert caplog.records, "the adapter should still log that it synthesised something"
    assert "emotion=" not in caplog.text
    if label.strip().lower() != "neutral":  # 'neutral' is also the fallback the log may name
        assert not _words(label).search(caplog.text)


async def test_espeak_rate_and_volume_are_not_logged_at_info(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A rate and a volume in wpm and 0-1 map one-to-one onto the emotion table.
    tts = _espeak(monkeypatch)

    with caplog.at_level(logging.INFO):
        await tts.synthesize("Hello there", emotion="angry")

    assert not re.search(r"\b(rate|volume|wpm)\b", caplog.text, re.IGNORECASE)


class TestNormalizeEmotionLogging:
    def test_an_unknown_label_is_warned_about_without_naming_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="intent_engine.tts.base"):
            assert normalize_emotion("overwhelmed") == "neutral"

        loud = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
        assert any("neutral" in message for message in loud)
        assert all("overwhelmed" not in message for message in loud)
        assert any("overwhelmed" in r.getMessage() for r in caplog.records)

    def test_the_label_is_bounded_in_the_debug_log(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG, logger="intent_engine.tts.base"):
            normalize_emotion("x" * 5000)

        assert len(caplog.text) < 500

    def test_a_non_string_is_reported_by_type_only(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="intent_engine.tts.base"):
            normalize_emotion(["angry"])

        assert "list" in caplog.text
        assert "angry" not in caplog.text

