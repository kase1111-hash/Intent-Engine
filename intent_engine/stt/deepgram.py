"""Deepgram STT adapter.

Uses the Deepgram SDK (``deepgram-sdk`` 5.x-7.x, ``AsyncDeepgramClient``)
to transcribe audio via the Deepgram API.
Requires a ``DEEPGRAM_API_KEY`` environment variable or explicit
``api_key`` parameter.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from prosody_protocol import ConversionError, WordAlignment
from prosody_protocol.alignment import from_deepgram

from intent_engine.errors import STTError
from intent_engine.stt.base import STTProvider, TranscriptionResult

logger = logging.getLogger(__name__)


class DeepgramSTT(STTProvider):
    """Deepgram API-based speech-to-text provider.

    Parameters
    ----------
    api_key:
        Deepgram API key.  Falls back to the ``DEEPGRAM_API_KEY``
        environment variable if not provided.
    model:
        Deepgram model to use (e.g., ``"nova-2"``).
    language:
        Language code (e.g., ``"en"``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "nova-2",
        language: str = "en",
        **kwargs: object,
    ) -> None:
        self._api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Deepgram API key is required. Set DEEPGRAM_API_KEY or pass api_key=."
            )
        self._model = model
        self._language = language

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio via the Deepgram API.

        Parameters
        ----------
        audio_path:
            Path to the audio file.

        Returns
        -------
        TranscriptionResult
            Transcription with word-level ``WordAlignment`` timestamps.

        Raises
        ------
        ImportError
            If ``deepgram-sdk`` is not installed, or is a version without
            ``AsyncDeepgramClient`` (before 5.0).
        FileNotFoundError
            If *audio_path* does not exist.
        STTError
            If the request fails or the response has no usable transcript.
        """
        try:
            from deepgram import AsyncDeepgramClient
        except ImportError as exc:
            if isinstance(exc, ModuleNotFoundError) and exc.name == "deepgram":
                raise ImportError(
                    "deepgram-sdk is required for DeepgramSTT. "
                    "Install it with: pip install intent-engine[deepgram]"
                ) from exc
            raise ImportError(
                f"The installed deepgram-sdk cannot be used by DeepgramSTT ({exc}). "
                "It needs deepgram-sdk>=5,<8: pip install -U 'deepgram-sdk>=5,<8'"
            ) from exc
        # A dependency of every deepgram-sdk release that has AsyncDeepgramClient, so
        # it is installed whenever this line is reached; type checking without the
        # optional extras cannot see it.
        import httpx  # type: ignore[import-not-found,unused-ignore]

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio = await asyncio.to_thread(path.read_bytes)

        # The SDK client has no close(): the httpx client it builds for itself keeps its
        # keep-alive connections open until it is garbage collected.  So the adapter owns
        # the httpx client, one per call (a client keeps its connections bound to the
        # event loop that first used it, and this adapter can be awaited from a different
        # loop each time), and closes it on every path.  The timeout and redirect
        # settings are the SDK's own defaults, which it applies only to a client it builds.
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http_client:
            client = AsyncDeepgramClient(api_key=self._api_key, httpx_client=http_client)
            try:
                response = await client.listen.v1.media.transcribe_file(
                    request=audio,
                    model=self._model,
                    language=self._language,
                    smart_format=True,
                    utterances=True,
                    punctuate=True,
                )
            except Exception as exc:
                raise STTError(f"Deepgram request failed: {type(exc).__name__}: {exc}") from exc

        # A request accepted for callback delivery answers with a request id only.
        results = getattr(response, "results", None)
        if results is None:
            raise STTError(
                "Deepgram returned no transcription results "
                "(was the request accepted for asynchronous processing?)"
            )

        channels = getattr(results, "channels", None) or []
        channel = channels[0] if channels else None
        alternatives = getattr(channel, "alternatives", None) or []
        alternative = alternatives[0] if alternatives else None
        text: str = getattr(alternative, "transcript", None) or ""
        detected_language: str | None = getattr(channel, "detected_language", None)

        alignments: list[WordAlignment] = []
        if text:
            try:
                alignments = from_deepgram(response)
            except ConversionError as exc:
                raise STTError(f"Deepgram returned unusable word timings: {exc}") from exc

        logger.info(
            "Deepgram transcribed %d words (model=%s, language=%s)",
            len(alignments),
            self._model,
            detected_language or self._language,
        )

        return TranscriptionResult(
            text=text,
            alignments=alignments,
            language=detected_language or self._language,
        )
