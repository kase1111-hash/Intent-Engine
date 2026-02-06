"""Tests for the Audio dataclass."""

from __future__ import annotations

import tempfile
from pathlib import Path

from intent_engine.models.audio import Audio


class TestAudioConstruction:
    def test_basic_construction(self) -> None:
        audio = Audio(data=b"\x00\x01\x02\x03")
        assert audio.data == b"\x00\x01\x02\x03"
        assert audio.format == "wav"
        assert audio.sample_rate == 16000
        assert audio.duration is None
        assert audio.url is None

    def test_with_all_fields(self) -> None:
        audio = Audio(
            data=b"audio_bytes",
            format="mp3",
            sample_rate=44100,
            duration=2.5,
            url="https://example.com/audio.mp3",
        )
        assert audio.format == "mp3"
        assert audio.sample_rate == 44100
        assert audio.duration == 2.5
        assert audio.url == "https://example.com/audio.mp3"


class TestAudioSave:
    def test_save_to_file(self) -> None:
        data = b"RIFF\x00\x00\x00\x00WAVEfmt "
        audio = Audio(data=data)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        audio.save(path)
        assert Path(path).read_bytes() == data
        Path(path).unlink()

    def test_save_accepts_path_object(self) -> None:
        data = b"test_audio"
        audio = Audio(data=data)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.wav"
            audio.save(path)
            assert path.read_bytes() == data


class TestAudioLen:
    def test_len_returns_byte_count(self) -> None:
        audio = Audio(data=b"\x00" * 100)
        assert len(audio) == 100

    def test_empty_audio(self) -> None:
        audio = Audio(data=b"")
        assert len(audio) == 0


class TestAudioMutability:
    def test_is_mutable(self) -> None:
        audio = Audio(data=b"original")
        audio.data = b"updated"
        assert audio.data == b"updated"
        audio.duration = 1.0
        assert audio.duration == 1.0
