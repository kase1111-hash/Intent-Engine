"""pitch_variance is speaker-relative pitch movement, not absolute Hz (audit #41).

Prosody Protocol 0.1.0a3 measures pitch relative to the speaker.  The same
melodic movement spans more Hz for a higher voice, so a Hz threshold pushed
high-pitched speakers out of ``pitch_variance: low`` for ordinary intonation.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from prosody_protocol import SpanFeatures

from intent_engine.constitutional.evaluator import (
    PITCH_VARIANCE_THRESHOLDS,
    check_required_prosody,
)
from intent_engine.constitutional.rules import PITCH_VARIANCE_LEVELS, ProsodyCondition

REGISTERS_HZ = [120.0, 220.0, 300.0, 350.0]


def _word(f0_base: float, semitones: float, *, samples: int = 40) -> SpanFeatures:
    """A word whose pitch swings +-``semitones`` around ``f0_base`` (10 ms samples)."""
    contour = [
        round(f0_base * 2 ** (semitones * math.sin(2 * math.pi * k / samples) / 12), 1)
        for k in range(samples)
    ]
    return SpanFeatures(
        start_ms=0,
        end_ms=samples * 10,
        text="w",
        f0_mean=f0_base,
        f0_range=(min(contour), max(contour)),
        f0_contour=contour,
        speech_rate=4.0,
    )


def _passes(level: str, features: list[SpanFeatures]) -> bool:
    passed, _ = check_required_prosody(ProsodyCondition(pitch_variance=level), features)
    return passed


class TestRegisterIndependence:
    @pytest.mark.parametrize("f0", REGISTERS_HZ)
    def test_mild_intonation_is_low_for_every_register(self, f0: float) -> None:
        assert _passes("low", [_word(f0, 1.0)])

    @pytest.mark.parametrize("f0", REGISTERS_HZ)
    def test_lively_intonation_is_high_for_every_register(self, f0: float) -> None:
        features = [_word(f0, 5.0)]
        assert _passes("high", features)
        assert not _passes("low", features)
        assert not _passes("normal", features)

    @pytest.mark.parametrize("f0", REGISTERS_HZ)
    def test_moderate_intonation_is_normal_for_every_register(self, f0: float) -> None:
        features = [_word(f0, 3.0)]
        assert _passes("normal", features)
        assert not _passes("low", features)
        assert not _passes("high", features)

    def test_same_movement_gets_the_same_label_in_every_register(self) -> None:
        for semitones in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
            labels = {
                tuple(_passes(level, [_word(f0, semitones)]) for level in PITCH_VARIANCE_LEVELS)
                for f0 in REGISTERS_HZ
            }
            assert len(labels) == 1, semitones

    def test_labels_partition_the_range(self) -> None:
        for semitones in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0):
            matches = [_passes(level, [_word(200.0, semitones)]) for level in PITCH_VARIANCE_LEVELS]
            assert sum(matches) == 1, semitones


class TestMeasure:
    def test_a_stray_sample_does_not_make_a_flat_voice_lively(self) -> None:
        contour = [120.0] * 39 + [240.0]  # one octave-jump artefact
        word = SpanFeatures(
            0, 400, "w", f0_range=(120.0, 240.0), f0_contour=contour, speech_rate=4.0
        )
        assert _passes("low", [word])

    def test_short_contour_falls_back_to_the_f0_range(self) -> None:
        flat = SpanFeatures(0, 40, "w", f0_range=(200.0, 204.0), f0_contour=[200.0, 204.0])
        wide = SpanFeatures(0, 40, "w", f0_range=(200.0, 400.0), f0_contour=[200.0, 400.0])
        assert _passes("low", [flat])
        assert _passes("high", [wide])

    def test_f0_range_alone_is_measured_in_semitones(self) -> None:
        """No contour: (100, 110) Hz and (300, 330) Hz are the same interval."""
        for lo, hi in ((100.0, 110.0), (300.0, 330.0)):
            assert _passes("low", [SpanFeatures(0, 100, "w", f0_range=(lo, hi))])
        for lo, hi in ((100.0, 200.0), (300.0, 600.0)):
            assert _passes("high", [SpanFeatures(0, 100, "w", f0_range=(lo, hi))])

    def test_spans_are_averaged(self) -> None:
        features = [_word(150.0, 0.5), _word(300.0, 5.0)]
        # 1 st and 9.5 st spreads average to about 5 st: normal.
        assert _passes("normal", features)

    def test_unvoiced_spans_are_skipped(self) -> None:
        features = [SpanFeatures(0, 100, "s"), _word(200.0, 1.0)]
        assert _passes("low", features)

    def test_reason_reports_semitones(self) -> None:
        passed, reason = check_required_prosody(
            ProsodyCondition(pitch_variance="low"), [_word(200.0, 5.0)]
        )
        assert passed is False
        assert reason is not None
        assert "semitones" in reason
        assert "Hz" not in reason


class TestThresholds:
    def test_levels_match_the_rule_schema(self) -> None:
        assert set(PITCH_VARIANCE_THRESHOLDS) == set(PITCH_VARIANCE_LEVELS)

    def test_levels_are_contiguous(self) -> None:
        bounds = [PITCH_VARIANCE_THRESHOLDS[level] for level in ("low", "normal", "high")]
        assert bounds[0][0] == 0.0
        assert bounds[0][1] == bounds[1][0]
        assert bounds[1][1] == bounds[2][0]
        assert bounds[2][1] == float("inf")


class TestRealAnalysis:
    """The same synthetic delivery through the real analyzer, at four registers."""

    @staticmethod
    def _analyse(tmp_path: Path, f0_base: float, semitones: float) -> list[SpanFeatures]:
        np = pytest.importorskip("numpy")
        pytest.importorskip("parselmouth")
        import wave

        from prosody_protocol import ProsodyAnalyzer, WordAlignment

        rate, word_s, gap_s = 16000, 0.5, 0.15
        pieces = [np.zeros(int(0.3 * rate))]
        alignments = []
        clock = 0.3
        for index in range(8):
            t = np.arange(int(word_s * rate)) / rate
            swing = semitones * np.sin(2 * np.pi * (t / word_s + 0.3 * index))
            phase = 2 * np.pi * np.cumsum(f0_base * 2 ** (swing / 12)) / rate
            voice = sum(np.sin(k * phase) / k for k in range(1, 25))
            envelope = np.minimum(1.0, np.minimum(t, word_s - t) / 0.03)
            pieces += [0.3 * voice / np.max(np.abs(voice)) * envelope, np.zeros(int(gap_s * rate))]
            alignments.append(
                WordAlignment(f"w{index}", int(clock * 1000), int((clock + word_s) * 1000))
            )
            clock += word_s + gap_s
        pieces.append(np.zeros(int(0.3 * rate)))
        pcm = (np.clip(np.concatenate(pieces), -1, 1) * 32767).astype("<i2")
        path = tmp_path / f"voice_{int(f0_base)}_{semitones}.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes(pcm.tobytes())
        return ProsodyAnalyzer().analyze(path, alignments)

    @pytest.mark.parametrize("f0", REGISTERS_HZ)
    def test_label_does_not_depend_on_the_register(self, tmp_path: Path, f0: float) -> None:
        mild = self._analyse(tmp_path, f0, 1.0)
        lively = self._analyse(tmp_path, f0, 5.0)
        assert _passes("low", mild)
        assert not _passes("low", lively)
        assert _passes("high", lively)
