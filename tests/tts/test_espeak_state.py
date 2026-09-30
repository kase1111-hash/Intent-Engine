"""ESpeakTTS must not leak voice, rate or volume from one call (or instance) to the next.

eSpeak's settings are process-wide and ``pyttsx3.init()`` returns the live engine
while anything still holds it, so an instance that only sets what it was
configured with inherits the rest from whichever call ran before it.  These tests
use ``fake_pyttsx3``, which models that, and ``test_espeak_real.py`` repeats the
check against the real library where it is installed.
"""

from __future__ import annotations

import asyncio
import sys
import threading

import pytest

from intent_engine.tts import espeak
from intent_engine.tts.espeak import ESpeakTTS
from tests.tts.fake_pyttsx3 import DEFAULT_VOICE, install_fake_pyttsx3


class TestVoiceIsSetOnEveryCall:
    async def test_default_voice_instance_does_not_inherit_another_instances_voice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        process = install_fake_pyttsx3(monkeypatch, keep_engines=True)
        female, default = ESpeakTTS(voice="en+f3"), ESpeakTTS()

        await female.synthesize("first")
        await default.synthesize("second")
        await female.synthesize("third")
        await default.synthesize("fourth")

        assert process.voices_by_text() == {
            "first": "en+f3",
            "second": DEFAULT_VOICE,
            "third": "en+f3",
            "fourth": DEFAULT_VOICE,
        }

    @pytest.mark.parametrize("keep_engines", [True, False])
    async def test_interleaved_instances_each_speak_in_their_own_voice(
        self, monkeypatch: pytest.MonkeyPatch, keep_engines: bool
    ) -> None:
        process = install_fake_pyttsx3(monkeypatch, keep_engines=keep_engines)
        instances = {"en+f3": ESpeakTTS(voice="en+f3"), "en+m7": ESpeakTTS(voice="en+m7")}
        instances[DEFAULT_VOICE] = ESpeakTTS()

        calls = [
            (f"{voice} {n}", tts)
            for n in range(8)
            for voice, tts in instances.items()
        ]
        await asyncio.gather(*(tts.synthesize(text) for text, tts in calls))

        wanted = {text: text.rsplit(" ", 1)[0] for text, _ in calls}
        assert process.voices_by_text() == wanted

    async def test_rate_and_volume_are_set_from_the_instance_every_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        process = install_fake_pyttsx3(monkeypatch, keep_engines=True)
        slow_quiet = ESpeakTTS(rate_wpm=100, volume=0.5)
        fast_loud = ESpeakTTS(rate_wpm=250, volume=1.0)

        await slow_quiet.synthesize("slow")
        await fast_loud.synthesize("fast")
        await slow_quiet.synthesize("slow again")

        rates = {synth.text: synth.rate for synth in process.synths}
        assert rates == {"slow": 100, "fast": 250, "slow again": 100}
        volumes = {synth.text: synth.volume for synth in process.synths}
        assert volumes["slow"] == volumes["slow again"] < volumes["fast"]

    async def test_a_cancelled_call_does_not_disturb_the_next_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cancelling stops the wait, not the synthesis; the next call is unaffected."""
        process = install_fake_pyttsx3(monkeypatch, keep_engines=True)
        process.hold = threading.Event()
        female, default = ESpeakTTS(voice="en+f3", rate_wpm=120), ESpeakTTS()

        cancelled = asyncio.create_task(female.synthesize("cancelled"))
        while not process.started.is_set():
            await asyncio.sleep(0.001)
        following = asyncio.create_task(default.synthesize("following"))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        process.hold.set()

        result = await asyncio.wait_for(following, 5)

        assert result.audio_data
        assert [(s.text, s.voice) for s in process.synths] == [
            ("cancelled", "en+f3"),
            ("following", DEFAULT_VOICE),
        ]
        assert process.synths[1].rate == 175


class TestEngineLifetime:
    async def test_the_engine_is_released_while_the_lock_is_held(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Destroying a pyttsx3 engine clears eSpeak's callback; never do it mid-synthesis."""
        process = install_fake_pyttsx3(monkeypatch)

        await ESpeakTTS().synthesize("hello")

        assert process.destroyed_while_locked == [True]

    async def test_the_next_call_starts_from_a_fresh_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        process = install_fake_pyttsx3(monkeypatch)

        await ESpeakTTS(voice="en+f3").synthesize("one")
        await ESpeakTTS().synthesize("two")

        assert process.destroyed_while_locked == [True, True]
        assert process.voices_by_text() == {"one": "en+f3", "two": DEFAULT_VOICE}


class TestDefaultVoiceLookup:
    def test_reads_the_driver_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_pyttsx3(monkeypatch)
        assert espeak._default_voice() == DEFAULT_VOICE

    def test_a_pyttsx3_without_the_espeak_driver_has_no_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_pyttsx3(monkeypatch)
        monkeypatch.setitem(sys.modules, "pyttsx3.drivers.espeak", None)
        assert espeak._default_voice() is None

    @pytest.mark.parametrize("value", ["", None, 7, object()])
    def test_an_unusable_default_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch, value: object
    ) -> None:
        install_fake_pyttsx3(monkeypatch)
        monkeypatch.setattr(
            sys.modules["pyttsx3.drivers.espeak"].EspeakDriver, "_defaultVoice", value
        )
        assert espeak._default_voice() is None

    async def test_a_configured_voice_wins_over_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        process = install_fake_pyttsx3(monkeypatch)
        await ESpeakTTS(voice="en+f3").synthesize("hello")
        assert process.synths[0].voice == "en+f3"
