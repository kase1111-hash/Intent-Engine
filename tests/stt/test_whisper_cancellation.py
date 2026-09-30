"""Cancelling a WhisperSTT call stops the wait, not the decode, and changes nothing after it.

A worker thread cannot be interrupted, so the decode that was running finishes
under the adapter's lock and its result is discarded; the next call must still
get the transcript of its own audio.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from intent_engine.stt.whisper import WhisperSTT

# Only bounds how long a failing test waits; a passing test never reaches it.
_STUCK_S = 5.0


def _result(text: str) -> dict[str, Any]:
    return {
        "text": f" {text}",
        "language": "en",
        "segments": [
            {"id": 0, "text": f" {text}", "words": [{"word": f" {text}", "start": 0.0, "end": 0.5}]}
        ],
    }


async def test_a_cancelled_transcription_does_not_disturb_the_next_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first_running = threading.Event()
    release_first = threading.Event()
    decoded: list[str] = []

    def transcribe(model: object, path: str, **kwargs: object) -> dict[str, Any]:
        name = Path(path).stem
        decoded.append(name)
        if name == "first":
            first_running.set()
            assert release_first.wait(_STUCK_S), "the decode was never released"
        return _result(name)

    module = types.ModuleType("whisper")
    module.load_model = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    module.transcribe = transcribe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "whisper", module)
    for name in ("first", "second"):
        (tmp_path / f"{name}.wav").write_bytes(b"audio")
    stt = WhisperSTT()

    first = asyncio.create_task(stt.transcribe(str(tmp_path / "first.wav")))
    while not first_running.is_set():
        await asyncio.sleep(0.001)
    second = asyncio.create_task(stt.transcribe(str(tmp_path / "second.wav")))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release_first.set()

    result = await asyncio.wait_for(second, _STUCK_S)

    assert result.text == "second"
    assert [a.word for a in result.alignments] == ["second"]
    assert decoded == ["first", "second"]
