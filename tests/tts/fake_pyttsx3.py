"""A pyttsx3 stand-in that keeps eSpeak's process-wide state, as the real one does.

Three things about the real stack matter to ``ESpeakTTS`` and are copied here:

* eSpeak's voice, rate and volume are **process-wide**: every engine object sets
  them on the same library, so an engine that skips setting one inherits whatever
  the last synthesis left behind.
* ``pyttsx3.init()`` hands out the **live** engine while any reference to it
  exists (it keeps engines in a ``WeakValueDictionary``), and only a *new* engine
  runs the driver constructor that resets voice, rate and volume.
* Destroying an engine clears eSpeak's synthesis callback, which must not happen
  while another call is synthesising.
"""

from __future__ import annotations

import sys
import threading
import types
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from intent_engine.tts import espeak
from tests.tts.helpers import wav_bytes

DEFAULT_VOICE = "gmw/en"


@dataclass
class Synth:
    """What eSpeak was set to when it synthesised ``text``."""

    text: str
    voice: str
    rate: int
    volume: float


@dataclass
class FakeEspeak:
    """The state shared by every fake engine, and a log of what happened to it."""

    voice: str = DEFAULT_VOICE
    rate: int = 200
    volume: float = 1.0
    synths: list[Synth] = field(default_factory=list)
    # For each engine destroyed: whether ESpeakTTS's engine lock was held at the time.
    destroyed_while_locked: list[bool] = field(default_factory=list)
    # Set by runAndWait() when it starts; if ``hold`` is set it then waits for it.
    started: threading.Event = field(default_factory=threading.Event)
    hold: threading.Event | None = None
    kept: list[Any] = field(default_factory=list)

    def voices_by_text(self) -> dict[str, str]:
        return {synth.text: synth.voice for synth in self.synths}


class _Engine:
    def __init__(self, process: FakeEspeak) -> None:
        self._process = process
        self._saved: tuple[str, str] | None = None
        self._callbacks: list[tuple[str, Any]] = []
        # The driver constructor resets eSpeak to its defaults.
        process.voice = DEFAULT_VOICE
        process.rate = 200
        process.volume = 1.0

    def __del__(self) -> None:
        self._process.destroyed_while_locked.append(espeak._ENGINE_LOCK.locked())

    def setProperty(self, name: str, value: Any) -> None:
        setattr(self._process, name, value)

    def save_to_file(self, text: str, path: str) -> None:
        self._saved = (text, path)

    def connect(self, topic: str, callback: Any) -> tuple[str, Any]:
        token = (topic, callback)
        self._callbacks.append(token)
        return token

    def disconnect(self, token: tuple[str, Any]) -> None:
        self._callbacks.remove(token)

    def runAndWait(self) -> None:
        assert self._saved is not None
        text, path = self._saved
        process = self._process
        process.started.set()
        if process.hold is not None:
            assert process.hold.wait(5), "the fake engine was never released"
        process.synths.append(Synth(text, process.voice, process.rate, process.volume))
        Path(path).write_bytes(wav_bytes())
        for topic, callback in list(self._callbacks):
            if topic == "finished-utterance":
                callback(name=None, completed=True)


def install_fake_pyttsx3(
    monkeypatch: pytest.MonkeyPatch, *, keep_engines: bool = False
) -> FakeEspeak:
    """Replace ``pyttsx3`` with a package whose eSpeak driver is :class:`FakeEspeak`.

    With ``keep_engines`` every engine stays referenced for the whole test, so
    ``init()`` keeps returning the first one.  That is what happens in real use
    when a call still holds the engine it used (or an exception's traceback
    does) while the next call starts.
    """
    process = FakeEspeak()
    active: weakref.WeakValueDictionary[None, _Engine] = weakref.WeakValueDictionary()

    def init(*args: object, **kwargs: object) -> _Engine:
        engine = active.get(None)
        if engine is None:
            engine = _Engine(process)
            active[None] = engine
            if keep_engines:
                process.kept.append(engine)
        return engine

    pyttsx3 = types.ModuleType("pyttsx3")
    pyttsx3.__path__ = []  # type: ignore[attr-defined]  # a package, so submodules import
    pyttsx3.init = init  # type: ignore[attr-defined]
    drivers = types.ModuleType("pyttsx3.drivers")
    drivers.__path__ = []  # type: ignore[attr-defined]
    driver = types.ModuleType("pyttsx3.drivers.espeak")

    class EspeakDriver:
        _defaultVoice = DEFAULT_VOICE

    driver.EspeakDriver = EspeakDriver  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyttsx3", pyttsx3)
    monkeypatch.setitem(sys.modules, "pyttsx3.drivers", drivers)
    monkeypatch.setitem(sys.modules, "pyttsx3.drivers.espeak", driver)
    return process
