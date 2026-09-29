"""type_to_speech hands plain text to TTS providers that do not read SSML.

No adapter processes SSML: they pass ``text`` straight to their engine, so
markup would be spoken (or billed) as if it were words.  Emotion reaches the
provider through the ``emotion`` argument.  A provider that does read SSML
opts in with ``supports_ssml = True``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from intent_engine.tts.base import SynthesisResult, TTSProvider
from tests.conftest import create_mocked_engine
from tests.tts.helpers import wav_bytes


class SpyTTS(TTSProvider):
    """Records what it is asked to synthesize."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def synthesize(
        self, text: str, emotion: str = "neutral", **kwargs: object
    ) -> SynthesisResult:
        self.calls.append((text, emotion))
        return SynthesisResult(audio_data=b"audio", format="wav", sample_rate=22050)


class SsmlSpyTTS(SpyTTS):
    supports_ssml = True


def _speak(tts: TTSProvider, text: str, emotion: str = "neutral") -> None:
    engine = create_mocked_engine()
    engine._tts = tts
    asyncio.run(engine.type_to_speech(text, emotion=emotion))


class TestPlainTextProviders:
    @pytest.mark.parametrize(
        "text",
        [
            "I'm feeling overwhelmed",
            "Hello, how are you today?",
            "a < b && c > d",
            "Tom & Jerry said \"hi\"",
            "<speak>already markup</speak>",
        ],
    )
    def test_the_typed_text_is_what_the_provider_receives(self, text: str) -> None:
        tts = SpyTTS()

        _speak(tts, text, emotion="sad")

        assert tts.calls == [(text, "sad")]

    def test_emotion_still_reaches_the_provider(self) -> None:
        tts = SpyTTS()

        _speak(tts, "Yes!", emotion="joyful")

        assert tts.calls[0][1] == "joyful"

    def test_non_core_emotion_is_passed_through_for_the_provider_to_map(self) -> None:
        tts = SpyTTS()

        _speak(tts, "Help", emotion="stressed")

        assert tts.calls == [("Help", "stressed")]

    def test_a_provider_that_only_looks_like_it_supports_ssml_gets_plain_text(self) -> None:
        tts = SpyTTS()
        tts.supports_ssml = False  # type: ignore[attr-defined]

        _speak(tts, "Hello there")

        assert tts.calls == [("Hello there", "neutral")]


class TestSsmlProviders:
    def test_a_provider_that_reads_ssml_gets_ssml(self) -> None:
        tts = SsmlSpyTTS()

        _speak(tts, "I am so happy!", emotion="joyful")

        text, emotion = tts.calls[0]
        assert text.startswith("<speak")
        assert "I am so happy!" in text
        assert emotion == "joyful"


class _FakePyttsx3Engine:
    """Stands in for pyttsx3's engine: records the text, writes a real WAV.

    Like pyttsx3, ``save_to_file()`` only records the request; the file is
    written and ``finished-utterance`` fired when ``runAndWait()`` runs.
    """

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.wav = wav_bytes()
        self._path: str | None = None
        self._finished: list[Any] = []

    def setProperty(self, name: str, value: Any) -> None:  # noqa: N802 - pyttsx3 API
        pass

    def connect(self, topic: str, callback: Any) -> Any:
        if topic == "finished-utterance":
            self._finished.append(callback)
        return (topic, callback)

    def disconnect(self, token: Any) -> None:
        pass

    def save_to_file(self, text: str, path: str) -> None:
        self.spoken.append(text)
        self._path = path

    def runAndWait(self) -> None:  # noqa: N802 - pyttsx3 API
        assert self._path is not None
        Path(self._path).write_bytes(self.wav)
        for callback in self._finished:
            callback(name=None, completed=True)


def test_the_real_espeak_adapter_is_asked_to_speak_the_sentence_not_markup() -> None:
    from intent_engine.tts.espeak import ESpeakTTS

    fake = _FakePyttsx3Engine()
    engine = create_mocked_engine()
    engine._tts = ESpeakTTS()

    with patch.object(ESpeakTTS, "_create_engine", return_value=fake):
        audio = asyncio.run(engine.type_to_speech("I'm feeling overwhelmed", emotion="sad"))

    assert fake.spoken == ["I'm feeling overwhelmed"]
    assert audio.data == fake.wav
