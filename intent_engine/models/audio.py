"""Audio dataclass -- output of IntentEngine.synthesize_speech()."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Audio:
    """Output of ``IntentEngine.synthesize_speech()``.

    Wraps raw audio bytes with metadata and provides convenience methods
    for saving to disk.
    """

    data: bytes
    """Raw audio bytes."""

    format: str = "wav"
    """Audio format (e.g., ``"wav"``, ``"mp3"``)."""

    sample_rate: int = 16000
    """Sample rate in Hz."""

    duration: float | None = None
    """Audio duration in seconds, or ``None`` if unknown."""

    url: str | None = None
    """Cloud-hosted URL for the audio, if applicable."""

    def save(self, path: str | Path) -> None:
        """Write audio bytes to a file.

        Parameters
        ----------
        path:
            Destination file path.
        """
        Path(path).write_bytes(self.data)

    def __len__(self) -> int:
        """Return the size of the audio data in bytes."""
        return len(self.data)
