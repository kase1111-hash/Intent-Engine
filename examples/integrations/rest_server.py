"""Generic REST API server exposing Intent Engine as HTTP endpoints.

Provides ``create_app()`` which returns a FastAPI application with
three endpoints:

- ``POST /process`` -- upload audio, returns transcription + emotion + IML
- ``POST /generate`` -- send IML, returns response text + emotion
- ``POST /synthesize`` -- send text + emotion, returns audio bytes

plus an unauthenticated ``GET /health``.

Usage (from the repository root)::

    # Providers and the API key come from the environment (see
    # create_app_from_env); the address defaults to 127.0.0.1:
    INTENT_API_KEY=change-me \\
        uvicorn --factory examples.integrations.rest_server:create_app_from_env

or, in your own module::

    from examples.integrations.rest_server import create_app

    app = create_app(stt_provider="whisper-prosody", llm_provider="claude", api_key="...")
    # Run with:  uvicorn your_module:app

Security: every endpoint spends provider quota (STT, LLM, TTS) and accepts
uploads.  This is a demo server.  Before making it reachable from anywhere
but your own machine, set an API key (``api_key=`` / ``INTENT_API_KEY``) and
put it behind TLS and a rate-limiting reverse proxy.  Request sizes are
bounded, but nothing here limits how many requests one client may make.
"""

import asyncio
import base64
import hmac
import importlib.metadata
import importlib.util
import logging
import os
import uuid
from collections.abc import Collection
from typing import Annotated, Any

from prosody_protocol import (
    AudioProcessingError,
    IMLParseError,
    IMLValidationError,
    IMLValidator,
)

from intent_engine.errors import LLMError, STTError, TTSError

from ._common import sniff_audio_suffix, temp_audio_file

logger = logging.getLogger(__name__)

DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
"""Largest audio upload ``create_app`` accepts by default (25 MiB)."""

DEFAULT_MAX_IML_CHARS = 200_000
"""Longest ``iml`` accepted by ``POST /generate`` by default."""

DEFAULT_MAX_TEXT_CHARS = 5_000
"""Longest ``text`` / ``context`` accepted by default (TTS engines cap around here)."""

_MAX_LABEL_CHARS = 64
_UPLOAD_CHUNK_BYTES = 64 * 1024
# Room for the multipart framing around an upload that is exactly at the limit.
_MULTIPART_OVERHEAD_BYTES = 64 * 1024
# A JSON encoder may write one character as up to 12 bytes (an escaped surrogate pair).
_JSON_BYTES_PER_CHAR = 12


def _package_version() -> str:
    try:
        return importlib.metadata.version("intent-engine")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def create_app(
    engine: Any = None,
    *,
    api_key: str | None = None,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    max_iml_chars: int = DEFAULT_MAX_IML_CHARS,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    **engine_kwargs: Any,
) -> Any:
    """Create a FastAPI application wrapping an Intent Engine instance.

    Parameters
    ----------
    engine:
        Pre-configured ``IntentEngine`` instance.  If ``None``, a new
        one is created from ``engine_kwargs``.
    api_key:
        If given, every endpoint except ``/health`` requires this value in
        an ``X-API-Key`` header (checked before the request body is read).
        Set one before exposing the server beyond localhost.
    max_upload_bytes:
        Largest audio upload; bigger ones get HTTP 413.
    max_iml_chars:
        Longest ``iml`` for ``/generate``.
    max_text_chars:
        Longest ``text`` for ``/synthesize`` and ``context`` for ``/generate``.
    **engine_kwargs:
        Keyword arguments forwarded to ``IntentEngine()`` if no
        ``engine`` is provided.

    Returns
    -------
    FastAPI
        A FastAPI application instance.  Each call returns an independent
        app.

    Notes
    -----
    Client mistakes (unsupported or undecodable audio, invalid IML) are
    answered with 415/422, failures of the STT/LLM/TTS providers with 502
    and anything else with 500.  Responses never contain exception text; the
    real cause is logged together with the ``error id`` the response quotes.
    """
    try:
        from fastapi import FastAPI, File, HTTPException, UploadFile
        from prosody_protocol.server.middleware import UploadSizeLimitMiddleware
        from pydantic import BaseModel, Field
        from starlette.datastructures import Headers
        from starlette.responses import JSONResponse
        from starlette.types import ASGIApp, Receive, Scope, Send
    except ImportError as exc:
        raise ImportError(
            "fastapi, pydantic and prosody-protocol's REST extra are required for the "
            "REST API server. Install them with: "
            "pip install fastapi uvicorn python-multipart httpx 'prosody-protocol[api]'"
        ) from exc

    # FastAPI only notices a missing python-multipart while registering the
    # upload route, with a RuntimeError that names no install command.
    if not any(importlib.util.find_spec(name) for name in ("python_multipart", "multipart")):
        raise ImportError(
            "python-multipart is required for audio uploads. "
            "Install it with: pip install python-multipart"
        )

    if api_key is not None and not api_key:
        raise ValueError("api_key must not be empty; pass None to disable authentication")

    if engine is None:
        from intent_engine.engine import IntentEngine
        engine = IntentEngine(**engine_kwargs)

    validator = IMLValidator()
    version = _package_version()

    # -- Request/Response models --

    class GenerateRequest(BaseModel):
        iml: str = Field(min_length=1, max_length=max_iml_chars)
        context: str | None = Field(default=None, max_length=max_text_chars)
        tone: str | None = Field(default=None, max_length=_MAX_LABEL_CHARS)

    class SynthesizeRequest(BaseModel):
        text: str = Field(min_length=1, max_length=max_text_chars)
        emotion: str = Field(default="neutral", max_length=_MAX_LABEL_CHARS)

    class ProcessResponse(BaseModel):
        text: str
        emotion: str
        confidence: float  # 0.0 (with "neutral") means no emotion was reported
        iml: str
        suggested_tone: str
        prosody_features: list[dict[str, Any]]

    class GenerateResponse(BaseModel):
        text: str
        emotion: str

    class SynthesizeResponse(BaseModel):
        audio_data: str  # base64-encoded
        format: str
        sample_rate: int
        duration: float | None = None

    class HealthResponse(BaseModel):
        status: str
        version: str

    # -- Middleware --

    class ApiKeyMiddleware:
        """Answer 401 unless the request carries the API key (``/health`` excepted).

        Plain ASGI middleware, so a request without the key is rejected
        before FastAPI reads (and spools) its body.
        """

        def __init__(
            self, app: ASGIApp, key: str, exempt_paths: Collection[str] = ("/health",)
        ) -> None:
            self.app = app
            self._key = key.encode()
            self._exempt = frozenset(exempt_paths)

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] == "http" and scope["path"] not in self._exempt:
                # Header values are decoded as latin-1; encoding them the same way
                # gives back the bytes the client sent.
                supplied = Headers(scope=scope).get("x-api-key", "").encode("latin-1")
                if not hmac.compare_digest(supplied, self._key):
                    response = JSONResponse(
                        {"detail": "Invalid or missing API key."}, status_code=401
                    )
                    await response(scope, receive, send)
                    return
            await self.app(scope, receive, send)

    def http_error(exc: Exception, action: str) -> HTTPException:
        """Map a pipeline failure to an HTTP error and log the real cause."""
        error_id = uuid.uuid4().hex[:12]
        if isinstance(exc, AudioProcessingError):
            status, message = 422, "The audio could not be decoded"
        elif isinstance(exc, (IMLParseError, IMLValidationError)):
            status, message = 422, "The IML is invalid"
        elif isinstance(exc, (STTError, LLMError, TTSError)):
            status, message = 502, "Upstream provider error"
        else:
            status, message = 500, "Internal server error"
        log = logger.error if status >= 500 else logger.warning
        log("Error %s [error id %s]", action, error_id, exc_info=exc)
        return HTTPException(status_code=status, detail=f"{message} (error id {error_id}).")

    # -- FastAPI app --

    api = FastAPI(
        title="Intent Engine API",
        description="Prosody-aware AI for emotional intelligence in voice conversations.",
        version=version,
    )

    # Added first, so it runs inside the authentication check below.
    max_json_bytes = _JSON_BYTES_PER_CHAR * (max_iml_chars + max_text_chars + _MAX_LABEL_CHARS)
    max_json_bytes += _MULTIPART_OVERHEAD_BYTES
    api.add_middleware(
        UploadSizeLimitMiddleware,
        max_bytes=max(max_upload_bytes + _MULTIPART_OVERHEAD_BYTES, max_json_bytes),
        max_json_bytes=max_json_bytes,
        upload_setting="max_upload_bytes",
        json_setting="max_iml_chars / max_text_chars",
    )
    if api_key is not None:
        api.add_middleware(ApiKeyMiddleware, key=api_key)

    @api.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="ok", version=version)

    @api.post("/process", response_model=ProcessResponse)
    async def process_audio(audio: Annotated[UploadFile, File()]) -> ProcessResponse:
        """Process an uploaded audio file through the full pipeline.

        Returns transcription, detected emotion, IML markup, and
        prosodic features.  The IML is validated by
        ``prosody_protocol.IMLValidator`` before being returned.  An
        ``emotion`` of ``"neutral"`` with ``confidence`` 0.0 means the
        engine reported no emotion.

        The audio type is detected from the content, never from the
        client-supplied file name.
        """
        chunks: list[bytes] = []
        size = 0
        while chunk := await audio.read(_UPLOAD_CHUNK_BYTES):
            size += len(chunk)
            if size > max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Audio upload exceeds the {max_upload_bytes} byte limit.",
                )
            chunks.append(chunk)
        contents = b"".join(chunks)

        if not contents:
            raise HTTPException(status_code=422, detail="Audio file is empty.")
        if sniff_audio_suffix(contents) is None:
            raise HTTPException(
                status_code=415,
                detail="Unsupported audio format. Send WAV, AIFF, FLAC, MP3, Ogg, WebM or M4A.",
            )

        try:
            async with temp_audio_file(contents) as tmp_path:
                result = await engine.process_voice_input(tmp_path)
        except Exception as exc:
            raise http_error(exc, "processing audio") from exc

        features = []
        for feat in result.prosody_features:
            feat_dict: dict[str, Any] = {
                "start_ms": getattr(feat, "start_ms", 0),
                "end_ms": getattr(feat, "end_ms", 0),
                "text": getattr(feat, "text", ""),
            }
            if getattr(feat, "f0_mean", None) is not None:
                feat_dict["f0_mean"] = feat.f0_mean
            if getattr(feat, "speech_rate", None) is not None:
                feat_dict["speech_rate"] = feat.speech_rate
            features.append(feat_dict)

        return ProcessResponse(
            text=result.text,
            emotion=result.emotion,
            confidence=result.confidence,
            iml=result.iml,
            suggested_tone=result.suggested_tone,
            prosody_features=features,
        )

    @api.post("/generate", response_model=GenerateResponse)
    async def generate_response(req: GenerateRequest) -> GenerateResponse:
        """Generate an LLM response from IML-annotated input."""
        validation = await asyncio.to_thread(validator.validate, req.iml)
        if not validation.valid:
            first = validation.errors[0].message[:200] if validation.errors else "invalid IML"
            raise HTTPException(status_code=422, detail=f"iml is not valid IML: {first}")

        try:
            response = await engine.generate_response(
                req.iml, context=req.context, tone=req.tone
            )
        except Exception as exc:
            raise http_error(exc, "generating response") from exc
        return GenerateResponse(text=response.text, emotion=response.emotion)

    @api.post("/synthesize", response_model=SynthesizeResponse)
    async def synthesize_speech(req: SynthesizeRequest) -> SynthesizeResponse:
        """Synthesize speech with emotional tone.

        Returns base64-encoded audio data with format metadata.
        """
        try:
            audio = await engine.synthesize_speech(req.text, emotion=req.emotion)
        except Exception as exc:
            raise http_error(exc, "synthesizing speech") from exc
        return SynthesizeResponse(
            audio_data=base64.b64encode(audio.data).decode(),
            format=audio.format,
            sample_rate=audio.sample_rate,
            duration=audio.duration,
        )

    return api


def create_app_from_env() -> Any:
    """Build the app from environment variables, for ``uvicorn --factory``.

    ============================  ==========================================
    ``INTENT_STT_PROVIDER``       ``stt_provider`` of ``IntentEngine``
    ``INTENT_LLM_PROVIDER``       ``llm_provider`` of ``IntentEngine``
    ``INTENT_TTS_PROVIDER``       ``tts_provider`` of ``IntentEngine``
    ``INTENT_API_KEY``            required ``X-API-Key`` value; unset means
                                  no authentication (a warning is logged)
    ============================  ==========================================

    Unset (or empty) provider variables keep the ``IntentEngine`` defaults;
    provider credentials are read by the provider adapters as usual.
    """
    engine_kwargs: dict[str, Any] = {
        kwarg: os.environ[var]
        for kwarg, var in (
            ("stt_provider", "INTENT_STT_PROVIDER"),
            ("llm_provider", "INTENT_LLM_PROVIDER"),
            ("tts_provider", "INTENT_TTS_PROVIDER"),
        )
        if os.environ.get(var)
    }
    api_key = os.environ.get("INTENT_API_KEY") or None
    if api_key is None:
        logger.warning(
            "INTENT_API_KEY is not set: the API is unauthenticated and spends provider "
            "quota for anyone who can reach it. Only run it on 127.0.0.1."
        )
    return create_app(api_key=api_key, **engine_kwargs)
