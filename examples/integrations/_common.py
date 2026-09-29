"""Helpers shared by the integration examples.

Kept deliberately small: a bounded, host-restricted media downloader, audio
type detection, a temp-file context manager for the engine (which takes a
file path), and the emotion-abstention check the chat/voice examples use
before showing an emotion to anyone.

Everything imported here is standard library; ``httpx`` is imported lazily
by :func:`download_media`.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path

from intent_engine.models.result import Result

MIN_EMOTION_CONFIDENCE = 0.5
"""Confidence below which an emotion is treated as "not reported".

The engine reports ``("neutral", 0.0)`` when the prosody analysis abstains,
and uses the same 0.5 threshold for ``Result.suggested_tone``.
"""

DEFAULT_MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
"""Default cap on a downloaded recording or attachment (25 MiB)."""


class MediaDownloadError(Exception):
    """A media download was refused (bad URL, too large) or failed."""


class UnsupportedAudioError(ValueError):
    """The bytes are not in an audio container the pipeline can read."""


def emotion_reported(result: Result) -> bool:
    """Whether *result* carries an emotion worth showing to a person.

    ``("neutral", 0.0)`` means the engine reported no emotion at all, so it
    must not be presented as a "neutral, 0% confidence" detection.
    """
    return result.confidence >= MIN_EMOTION_CONFIDENCE


def sniff_audio_suffix(data: bytes) -> str | None:
    """Return a file suffix for the audio container *data* starts with.

    Detects WAV, AIFF, FLAC, Ogg, MP3, Matroska/WebM and MP4/M4A by their
    magic bytes; returns ``None`` for anything else (an HTML login page, a
    text file, an empty upload, ...).  The suffix is only a label: the
    decoders read the content, and ffmpeg is needed for the container types
    ``prosody_protocol`` cannot read natively (Ogg/Opus, WebM, M4A).
    """
    head = data[:12]
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return ".wav"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return ".aiff"
    if head[:4] == b"fLaC":
        return ".flac"
    if head[:4] == b"OggS":
        return ".ogg"
    if head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0):
        return ".mp3"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    if head[4:8] == b"ftyp":
        return ".m4a"
    return None


@asynccontextmanager
async def temp_audio_file(data: bytes) -> AsyncIterator[str]:
    """Write *data* to a temporary audio file and yield its path.

    The suffix comes from the content, never from a client-supplied name.
    The file is written off the event loop and removed on exit, including
    when the body raises.

    Raises
    ------
    UnsupportedAudioError
        If *data* is not a recognised audio container.
    """
    suffix = sniff_audio_suffix(data)
    if suffix is None:
        raise UnsupportedAudioError("Unrecognised audio format")
    with tempfile.TemporaryDirectory(prefix="intent-engine-") as tmp_dir:
        path = Path(tmp_dir) / f"audio{suffix}"
        await asyncio.to_thread(path.write_bytes, data)
        yield str(path)


async def download_media(
    url: str,
    *,
    allowed_hosts: Iterable[str],
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    headers: Mapping[str, str] | None = None,
    timeout: float = 30.0,
) -> bytes:
    """Download *url* into memory, refusing hosts and sizes it should not fetch.

    Only ``https`` URLs whose host is one of *allowed_hosts* (or a subdomain
    of one) are fetched, so credentials in *headers* are never sent
    anywhere else and a forged URL cannot be used to probe internal
    services.  Redirects are not followed.  The body is streamed and
    abandoned as soon as it exceeds *max_bytes*.

    Raises
    ------
    MediaDownloadError
        If the URL is not allowed, the response is too large or not a 2xx,
        or the request fails.  The message never includes the URL.
    """
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required for downloading media. Install it with: pip install httpx"
        ) from exc

    # Parse with httpx itself, the parser that will make the request, so the
    # host checked is the host contacted.
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL as exc:
        raise MediaDownloadError("URL is not allowed") from exc
    host = parsed.host.lower()
    if parsed.scheme != "https" or not any(
        host == allowed or host.endswith("." + allowed) for allowed in allowed_hosts
    ):
        raise MediaDownloadError("URL is not an https URL on an allowed host")

    chunks: list[bytes] = []
    size = 0
    try:
        async with (
            httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client,
            client.stream("GET", url, headers=dict(headers or {})) as resp,
        ):
            resp.raise_for_status()
            declared = resp.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > max_bytes:
                raise MediaDownloadError(f"Download too large (limit {max_bytes} bytes)")
            async for chunk in resp.aiter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise MediaDownloadError(f"Download too large (limit {max_bytes} bytes)")
                chunks.append(chunk)
    except httpx.HTTPStatusError as exc:
        raise MediaDownloadError(
            f"Download failed with HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        raise MediaDownloadError(f"Download failed ({type(exc).__name__})") from exc
    return b"".join(chunks)
