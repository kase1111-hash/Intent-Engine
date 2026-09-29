"""ESpeakTTS against the real pyttsx3 + eSpeak stack.

Skipped unless pyttsx3 (2.99 or later) and a working eSpeak / espeak-ng
library are installed, which the CI environment does not have.  Where they
are, these tests check what the stubbed tests in ``test_espeak.py`` cannot:
that longer text and the emotion-to-volume mapping hold up with real audio.
"""

from __future__ import annotations

import asyncio
import io
import struct
import sys
import wave

import pytest

from intent_engine.tts import espeak
from intent_engine.tts.espeak import ESpeakTTS


@pytest.fixture(scope="module", autouse=True)
def real_espeak() -> None:
    pytest.importorskip("pyttsx3")
    if not sys.platform.startswith("linux"):
        pytest.skip("only the Linux eSpeak driver is exercised")
    version = espeak._pyttsx3_version()
    if version is None or version < (2, 99):
        pytest.skip("needs pyttsx3 2.99 or later (older versions need the ffmpeg binary)")
    import pyttsx3

    try:
        pyttsx3.init()
    except Exception as exc:  # no libespeak-ng, or no usable voice
        pytest.skip(f"eSpeak is not usable here: {exc}")


def _wav_stats(audio: bytes) -> tuple[int, int]:
    """Return ``(frames, peak amplitude)`` of a 16-bit mono WAV."""
    with wave.open(io.BytesIO(audio), "rb") as wf:
        frames = wf.getnframes()
        raw = wf.readframes(frames)
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    return frames, max((abs(s) for s in samples), default=0)


async def test_short_sentence_yields_audio() -> None:
    result = await ESpeakTTS().synthesize("Hello there, how are you today?")
    frames, peak = _wav_stats(result.audio_data)
    assert frames > 0
    assert peak > 0


async def test_long_text_is_not_cut_short_or_empty() -> None:
    # pyttsx3 2.99 returns from runAndWait() before eSpeak has finished, which
    # used to give an empty file for anything longer than a short sentence.
    text = " ".join(["This sentence goes on and on to make the utterance long."] * 12)
    short = await ESpeakTTS().synthesize(text[:60])
    long = await ESpeakTTS().synthesize(text)
    assert _wav_stats(long.audio_data)[0] > 5 * _wav_stats(short.audio_data)[0]


async def test_louder_emotions_are_measurably_louder() -> None:
    tts = ESpeakTTS()
    peaks = {}
    for emotion in ("sad", "neutral", "frustrated", "angry"):
        result = await tts.synthesize("Hello there, how are you today?", emotion=emotion)
        peaks[emotion] = _wav_stats(result.audio_data)[1]
    assert peaks["sad"] < peaks["neutral"] < peaks["frustrated"] < peaks["angry"]


async def test_ssml_is_not_read_aloud() -> None:
    plain = await ESpeakTTS().synthesize("I am so happy!")
    marked_up = await ESpeakTTS().synthesize(
        '<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        '<s><prosody pitch="+5%" volume="+3dB">I am so happy!</prosody></s></speak>'
    )
    assert abs(_wav_stats(marked_up.audio_data)[0] - _wav_stats(plain.audio_data)[0]) < 1000


async def test_concurrent_calls_all_produce_audio() -> None:
    results = await asyncio.gather(
        *(ESpeakTTS().synthesize(f"Sentence number {i}.", emotion="calm") for i in range(4))
    )
    assert all(_wav_stats(r.audio_data)[0] > 0 for r in results)
