"""Helpers shared by the TTS adapter tests."""

from __future__ import annotations

import asyncio
import io
import wave
from types import TracebackType

# Every value of the ``output_format`` Literal in elevenlabs 2.70 and the
# (format, sample rate) an ElevenLabsTTS reports for it.
SDK_OUTPUT_FORMATS = {
    "alaw_8000": ("alaw", 8000),
    "mp3_22050_32": ("mp3", 22050),
    "mp3_24000_48": ("mp3", 24000),
    "mp3_44100_128": ("mp3", 44100),
    "mp3_44100_192": ("mp3", 44100),
    "mp3_44100_32": ("mp3", 44100),
    "mp3_44100_64": ("mp3", 44100),
    "mp3_44100_96": ("mp3", 44100),
    "opus_48000_128": ("opus", 48000),
    "opus_48000_192": ("opus", 48000),
    "opus_48000_32": ("opus", 48000),
    "opus_48000_64": ("opus", 48000),
    "opus_48000_96": ("opus", 48000),
    "pcm_16000": ("pcm", 16000),
    "pcm_22050": ("pcm", 22050),
    "pcm_24000": ("pcm", 24000),
    "pcm_32000": ("pcm", 32000),
    "pcm_44100": ("pcm", 44100),
    "pcm_48000": ("pcm", 48000),
    "pcm_8000": ("pcm", 8000),
    "ulaw_8000": ("ulaw", 8000),
    "wav_16000": ("wav", 16000),
    "wav_22050": ("wav", 22050),
    "wav_24000": ("wav", 24000),
    "wav_32000": ("wav", 32000),
    "wav_44100": ("wav", 44100),
    "wav_48000": ("wav", 48000),
    "wav_8000": ("wav", 8000),
}


def wav_bytes(frames: int = 2205, rate: int = 22050) -> bytes:
    """Return a small, valid mono 16-bit WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x01\x00" * frames)
    return buf.getvalue()


class Heartbeat:
    """Count event-loop ticks while a block of code runs.

    An adapter that blocks the loop leaves the heartbeat task starved, so
    the tick count stays near zero; one that hands the work to a thread
    lets it run about ``duration / interval`` times.
    """

    def __init__(self, interval: float = 0.01) -> None:
        self.interval = interval
        self.ticks = 0
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            self.ticks += 1

    async def __aenter__(self) -> Heartbeat:
        self._task = asyncio.create_task(self._run())
        await asyncio.sleep(0)  # let the task start before the timed block
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._task is not None
        self._task.cancel()
