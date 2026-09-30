"""ESpeakTTS against the real pyttsx3 + eSpeak stack.

Skipped unless pyttsx3 (2.99 or later) and a working eSpeak / espeak-ng
library are installed.  The ``sdk-contracts`` CI job installs both and sets
``REQUIRE_REAL_ESPEAK=1``, which turns every skip below into a failure so the
job cannot quietly stop running these tests.  Where they run, they check what
the stubbed tests in ``test_espeak.py`` cannot: that longer text and the
emotion-to-volume mapping hold up with real audio.
"""

from __future__ import annotations

import asyncio
import io
import os
import struct
import sys
import wave

import pytest

from intent_engine.tts import espeak
from intent_engine.tts.espeak import ESpeakTTS


def _unavailable(reason: str) -> None:
    """Skip, unless CI has said these tests must run."""
    if os.environ.get("REQUIRE_REAL_ESPEAK") == "1":
        pytest.fail(f"REQUIRE_REAL_ESPEAK=1 but the real eSpeak stack is unusable: {reason}")
    pytest.skip(reason)


@pytest.fixture(scope="module", autouse=True)
def real_espeak() -> None:
    try:
        import pyttsx3
    except ImportError:
        _unavailable("pyttsx3 is not installed")
        return
    if not sys.platform.startswith("linux"):
        _unavailable("only the Linux eSpeak driver is exercised")
    version = espeak._pyttsx3_version()
    if version is None or version < (2, 99):
        _unavailable("needs pyttsx3 2.99 or later (older versions need the ffmpeg binary)")

    try:
        pyttsx3.init()
    except Exception as exc:  # no libespeak-ng, or no usable voice
        _unavailable(f"eSpeak is not usable here: {exc}")


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


def _median_f0(audio: bytes) -> float:
    """Median fundamental frequency (Hz) of the voiced frames of a 16-bit mono WAV."""
    np = pytest.importorskip("numpy")
    with wave.open(io.BytesIO(audio), "rb") as wf:
        rate = wf.getframerate()
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(float)
    frame, hop = int(0.04 * rate), int(0.02 * rate)
    lags = np.arange(int(rate / 400), int(rate / 60))
    estimates = []
    for start in range(0, len(samples) - frame, hop):
        window = samples[start : start + frame]
        window = window - window.mean()
        if np.sqrt(np.mean(window**2)) < 500:  # unvoiced or silent
            continue
        corr = np.correlate(window, window, "full")[frame - 1 :]
        if corr[0] <= 0:
            continue
        best = lags[np.argmax(corr[lags])]
        if corr[best] > 0.5 * corr[0]:  # clearly periodic
            estimates.append(rate / best)
    assert estimates, "no voiced frames found"
    return float(np.median(estimates))


async def test_default_voice_instance_is_unaffected_by_another_instance() -> None:
    text = "Hello there, this is a test sentence."
    female, default = ESpeakTTS(voice="en+f3"), ESpeakTTS()
    alone = _median_f0((await default.synthesize(text)).audio_data)
    female_alone = _median_f0((await female.synthesize(text)).audio_data)
    assert female_alone > alone * 1.3, "the two voices should be easy to tell apart"

    results = await asyncio.gather(
        *(tts.synthesize(text) for _ in range(10) for tts in (female, default))
    )

    pitches = [_median_f0(r.audio_data) for r in results]
    assert all(abs(f0 - female_alone) < abs(f0 - alone) for f0 in pitches[0::2])
    assert all(abs(f0 - alone) < abs(f0 - female_alone) for f0 in pitches[1::2])
