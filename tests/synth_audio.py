"""Synthetic speech-like recordings for tests that run the real analyzer.

Each word is a harmonic tone with a chosen pitch, loudness, syllable rate
and optional pitch glide, so the real ``ProsodyAnalyzer`` and
``IMLAssembler`` see distinct, repeatable speaker behaviour without any
recorded speech.  Only the STT provider is faked (it returns the word
alignments that were used to build the audio).

Needs ``numpy`` (a dependency of ``prosody-protocol[audio]``); test modules
call ``pytest.importorskip("numpy")`` and ``pytest.importorskip("parselmouth")``
before importing this module.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
from prosody_protocol import WordAlignment

from intent_engine.engine import IntentEngine
from intent_engine.stt.base import TranscriptionResult
from tests.conftest import create_mocked_engine

SAMPLE_RATE = 16000

_WORDS = ["Well", "that", "is", "what", "we", "said."]

#: An ordinary, level delivery: it becomes the speaker baseline.
NEUTRAL: dict[str, float] = {"f0": 140, "gain_db": -20, "syl": 4.0}
#: Higher pitch with lively pitch movement (reads as joyful).
EXCITED: dict[str, float] = {"f0": 170, "gain_db": -18, "syl": 4.5, "glide": 8.0}
#: Louder, higher and faster with flat pitch (reads as angry).
ANGRY: dict[str, float] = {"f0": 190, "gain_db": -11, "syl": 5.5}
#: Lower, quieter and slower (reads as sad).
SAD: dict[str, float] = {"f0": 115, "gain_db": -25, "syl": 2.6}


def _word(dur: float, f0: float, gain_db: float, syl: float, glide_st: float) -> Any:
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    glide = np.linspace(-glide_st / 2, glide_st / 2, n)
    freq = f0 * 2 ** (glide / 12) * (1 + 0.004 * np.sin(2 * np.pi * 5 * t))
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    sig = sum(np.sin(h * phase) / h for h in range(1, 12))
    sig = sig * (0.55 + 0.45 * np.cos(2 * np.pi * syl * t - np.pi))
    sig = sig * np.minimum(1, np.minimum(t, dur - t) / 0.02)
    return sig / np.max(np.abs(sig)) * 10 ** (gain_db / 20)


def write_recording(path: Path, utterances: list[dict[str, float]]) -> list[WordAlignment]:
    """Write a WAV of one six-word sentence per entry of *utterances*.

    Returns the matching word alignments (what a real STT would report).
    """
    rng = np.random.default_rng(0)
    chunks: list[Any] = [np.zeros(int(0.3 * SAMPLE_RATE))]
    t_ms = 300.0
    aligns: list[WordAlignment] = []
    for u in utterances:
        for i, word in enumerate(_WORDS):
            chunks.append(_word(0.35, u["f0"], u["gain_db"], u["syl"], u.get("glide", 0.0)))
            aligns.append(WordAlignment(word, int(t_ms), int(t_ms + 350)))
            t_ms += 350
            gap = 0.8 if i == len(_WORDS) - 1 else 0.06
            chunks.append(np.zeros(int(gap * SAMPLE_RATE)))
            t_ms += gap * 1000
    chunks.append(np.zeros(int(0.4 * SAMPLE_RATE)))
    audio = np.concatenate(chunks)
    audio = audio + rng.normal(0, 10 ** (-70 / 20), audio.shape)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return aligns


def engine_for(aligns: list[WordAlignment], **kwargs: Any) -> IntentEngine:
    """An engine whose STT returns *aligns*; the analyzer and assembler are real."""
    engine = create_mocked_engine(**kwargs)
    engine._stt.transcribe = AsyncMock(
        return_value=TranscriptionResult(
            text=" ".join(a.word for a in aligns), alignments=list(aligns), language="en"
        )
    )
    return engine
