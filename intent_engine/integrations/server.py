"""Generic REST API server exposing Intent Engine as HTTP endpoints.

Provides ``create_app()`` which returns a FastAPI application with
three endpoints:

- ``POST /process`` -- upload audio, returns transcription + emotion + IML
- ``POST /generate`` -- send IML, returns response text + emotion
- ``POST /synthesize`` -- send text + emotion, returns audio bytes

Usage::

    from intent_engine.integrations.server import create_app

    app = create_app(stt_provider="whisper-prosody", llm_provider="claude")

    # Run with:  uvicorn intent_engine.integrations.server:app
"""

import base64
import logging
import tempfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Module-level app instance for ``uvicorn intent_engine.integrations.server:app``
app: Any = None


def create_app(
    engine: Any = None,
    **engine_kwargs: Any,
) -> Any:
    """Create a FastAPI application wrapping an Intent Engine instance.

    Parameters
    ----------
    engine:
        Pre-configured ``IntentEngine`` instance.  If ``None``, a new
        one is created from ``engine_kwargs``.
    **engine_kwargs:
        Keyword arguments forwarded to ``IntentEngine()`` if no
        ``engine`` is provided.

    Returns
    -------
    FastAPI
        A FastAPI application instance.
    """
    try:
        from fastapi import FastAPI, File, HTTPException, UploadFile
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "fastapi and pydantic are required for the REST API server. "
            "Install them with: pip install fastapi uvicorn pydantic"
        )

    if engine is None:
        from intent_engine.engine import IntentEngine
        engine = IntentEngine(**engine_kwargs)

    # -- Request/Response models --

    class GenerateRequest(BaseModel):
        iml: str
        context: str | None = None
        tone: str | None = None

    class SynthesizeRequest(BaseModel):
        text: str
        emotion: str = "neutral"

    class ProcessResponse(BaseModel):
        text: str
        emotion: str
        confidence: float
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

    # -- FastAPI app --

    api = FastAPI(
        title="Intent Engine API",
        description="Prosody-aware AI for emotional intelligence in voice conversations.",
        version="0.8.0",
    )

    @api.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="ok", version="0.8.0")

    @api.post("/process", response_model=ProcessResponse)
    async def process_audio(audio: UploadFile = File(...)) -> ProcessResponse:
        """Process an uploaded audio file through the full pipeline.

        Returns transcription, detected emotion, IML markup, and
        prosodic features.  The IML is validated by
        ``prosody_protocol.IMLValidator`` before being returned.
        """
        with tempfile.NamedTemporaryFile(
            suffix=_suffix_from_filename(audio.filename),
            delete=False,
        ) as tmp:
            contents = await audio.read()
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            result = await engine.process_voice_input(tmp_path)

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
        except Exception as exc:
            logger.exception("Error processing audio")
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @api.post("/generate", response_model=GenerateResponse)
    async def generate_response(req: GenerateRequest) -> GenerateResponse:
        """Generate an LLM response from IML-annotated input."""
        try:
            response = await engine.generate_response(
                req.iml, context=req.context, tone=req.tone
            )
            return GenerateResponse(text=response.text, emotion=response.emotion)
        except Exception as exc:
            logger.exception("Error generating response")
            raise HTTPException(status_code=500, detail=str(exc))

    @api.post("/synthesize", response_model=SynthesizeResponse)
    async def synthesize_speech(req: SynthesizeRequest) -> SynthesizeResponse:
        """Synthesize speech with emotional tone.

        Returns base64-encoded audio data with format metadata.
        """
        try:
            audio = await engine.synthesize_speech(req.text, emotion=req.emotion)
            return SynthesizeResponse(
                audio_data=base64.b64encode(audio.data).decode(),
                format=audio.format,
                sample_rate=audio.sample_rate,
                duration=audio.duration,
            )
        except Exception as exc:
            logger.exception("Error synthesizing speech")
            raise HTTPException(status_code=500, detail=str(exc))

    # Store reference at module level for uvicorn
    global app
    app = api

    return api


def _suffix_from_filename(filename: str | None) -> str:
    """Extract file suffix from an upload filename."""
    if filename:
        suffix = Path(filename).suffix
        if suffix:
            return suffix
    return ".wav"
