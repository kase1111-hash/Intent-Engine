"""Tests for the helpers shared by the integration examples."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from examples.integrations._common import (
    UnsupportedAudioError,
    emotion_reported,
    sniff_audio_suffix,
    temp_audio_file,
)

# -- Emotion abstention --


class TestEmotionReported:
    def test_abstention_is_not_a_detection(self, make_result: Callable[..., Any]) -> None:
        # ("neutral", 0.0) is what the engine returns when no emotion is reported.
        assert not emotion_reported(make_result(emotion="neutral", confidence=0.0))

    def test_confident_emotion_is_reported(self, make_result: Callable[..., Any]) -> None:
        assert emotion_reported(make_result(emotion="joyful", confidence=0.85))

    def test_confident_neutral_is_reported(self, make_result: Callable[..., Any]) -> None:
        assert emotion_reported(make_result(emotion="neutral", confidence=0.7))

    def test_low_confidence_is_not_reported(self, make_result: Callable[..., Any]) -> None:
        assert not emotion_reported(make_result(emotion="sad", confidence=0.3))

    def test_the_threshold_is_the_engines_own(self, make_result: Callable[..., Any]) -> None:
        # Result.suggested_tone switches to the emotion at 0.5, and so do the examples.
        assert emotion_reported(make_result(emotion="sad", confidence=0.5))
        assert not emotion_reported(make_result(emotion="sad", confidence=0.49))


# -- Audio type detection --


class TestSniffAudioSuffix:
    @pytest.mark.parametrize(
        ("head", "suffix"),
        [
            (b"RIFF\x24\x00\x00\x00WAVEfmt ", ".wav"),
            (b"FORM\x00\x00\x00\x00AIFFCOMM", ".aiff"),
            (b"fLaC\x00\x00\x00\x22", ".flac"),
            (b"OggS\x00\x02\x00\x00", ".ogg"),
            (b"ID3\x04\x00\x00\x00\x00", ".mp3"),
            (b"\xff\xfb\x90\x00", ".mp3"),
            (b"\x1a\x45\xdf\xa3\x01\x00", ".webm"),
            (b"\x00\x00\x00\x20ftypM4A ", ".m4a"),
        ],
    )
    def test_recognised_containers(self, head: bytes, suffix: str) -> None:
        assert sniff_audio_suffix(head + b"\x00" * 32) == suffix

    @pytest.mark.parametrize(
        "data",
        [b"", b"not audio at all", b"RIFF fake audio", b"<html>login</html>", b"\x00" * 64],
    )
    def test_unrecognised_content(self, data: bytes) -> None:
        assert sniff_audio_suffix(data) is None


# -- temp_audio_file --


class TestTempAudioFile:
    async def test_writes_file_with_detected_suffix(self, wav_bytes: bytes) -> None:
        async with temp_audio_file(wav_bytes) as path:
            assert path.endswith(".wav")
            assert Path(path).read_bytes() == wav_bytes

    async def test_file_removed_afterwards(self, wav_bytes: bytes) -> None:
        async with temp_audio_file(wav_bytes) as path:
            pass
        assert not Path(path).exists()

    async def test_file_removed_when_body_raises(self, wav_bytes: bytes) -> None:
        seen: list[str] = []
        with pytest.raises(RuntimeError):
            async with temp_audio_file(wav_bytes) as path:
                seen.append(path)
                raise RuntimeError("boom")
        assert not Path(seen[0]).exists()

    async def test_rejects_non_audio(self) -> None:
        with pytest.raises(UnsupportedAudioError):
            async with temp_audio_file(b"not audio"):
                pytest.fail("body must not run for unsupported audio")

    async def test_write_does_not_run_on_the_event_loop_thread(
        self, wav_bytes: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loop_thread = threading.get_ident()
        writers: list[int] = []
        real_write = Path.write_bytes

        def spy(self: Path, data: bytes) -> int:
            writers.append(threading.get_ident())
            return real_write(self, data)

        monkeypatch.setattr(Path, "write_bytes", spy)
        async with temp_audio_file(wav_bytes):
            pass
        assert writers
        assert loop_thread not in writers
