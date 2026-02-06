"""Twilio voice webhook handler.

Provides ``TwilioVoiceHandler``, a class that processes incoming
Twilio voice webhooks through the Intent Engine pipeline and
returns TwiML responses with emotionally-aware synthesized speech.

Usage::

    from intent_engine import IntentEngine
    from intent_engine.integrations.twilio import TwilioVoiceHandler

    engine = IntentEngine()
    handler = TwilioVoiceHandler(engine)

    # In a Flask/FastAPI route:
    twiml = await handler.handle_voice(recording_url, form_data)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Callable

from intent_engine.errors import IntentEngineError
from intent_engine.models.result import Result

logger = logging.getLogger(__name__)


class TwilioVoiceHandler:
    """Process Twilio voice webhooks through the Intent Engine pipeline.

    Receives a recording URL from Twilio, downloads the audio,
    runs it through ``IntentEngine.process_voice_input()``, generates
    a response, synthesizes speech, and returns TwiML XML.

    Parameters
    ----------
    engine:
        An ``IntentEngine`` (or deployment variant) instance.
    response_callback:
        Optional callback ``(Result) -> str`` that returns custom
        response text based on the pipeline result.  If not provided,
        a default handler generates a generic response.
    download_func:
        Optional async callable ``(url) -> bytes`` for downloading
        audio from Twilio.  Defaults to using ``httpx``.
    """

    def __init__(
        self,
        engine: Any,
        response_callback: Callable[[Result], str] | None = None,
        download_func: Callable[..., Any] | None = None,
    ) -> None:
        self._engine = engine
        self._response_callback = response_callback
        self._download_func = download_func

    async def _download_audio(self, url: str) -> bytes:
        """Download audio bytes from a URL."""
        if self._download_func:
            return await self._download_func(url)

        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for downloading Twilio recordings. "
                "Install it with: pip install httpx"
            )

        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    async def handle_voice(
        self,
        recording_url: str,
        form_data: dict[str, str] | None = None,
    ) -> str:
        """Process a Twilio voice webhook and return TwiML XML.

        Parameters
        ----------
        recording_url:
            URL of the Twilio recording to process.
        form_data:
            Optional Twilio webhook form data (``CallSid``, ``From``,
            etc.) for logging/context.

        Returns
        -------
        str
            TwiML XML response string.
        """
        call_sid = (form_data or {}).get("CallSid", "unknown")
        logger.info("Processing Twilio voice webhook (CallSid=%s)", call_sid)

        # Download audio to a temporary file
        audio_bytes = await self._download_audio(recording_url)

        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            # Process through the Intent Engine pipeline
            result = await self._engine.process_voice_input(tmp_path)

            # Generate response text
            if self._response_callback:
                response_text = self._response_callback(result)
            else:
                response_text = self._default_response(result)

            # Synthesize speech with emotion
            audio = await self._engine.synthesize_speech(
                response_text, emotion=result.suggested_tone
            )

            # Build TwiML response
            return self._build_twiml(audio_url=audio.url, text=response_text)

        except IntentEngineError:
            logger.exception("Intent Engine error processing call %s", call_sid)
            return self._build_twiml(
                text="I'm sorry, I'm having trouble processing your request. "
                     "Please try again.",
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @staticmethod
    def _default_response(result: Result) -> str:
        """Generate a default response based on emotion detection."""
        if result.emotion == "frustrated":
            return (
                "I can hear you're frustrated. "
                "Let me escalate this to a specialist who can help."
            )
        if result.emotion == "angry":
            return (
                "I understand your concern. "
                "Let me connect you with someone who can help resolve this."
            )
        return f"I heard: {result.text}. Let me help you with that."

    @staticmethod
    def _build_twiml(
        audio_url: str | None = None,
        text: str | None = None,
    ) -> str:
        """Build a TwiML XML response string.

        Uses ``<Play>`` if an audio URL is available, otherwise
        falls back to ``<Say>``.
        """
        parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<Response>"]
        if audio_url:
            parts.append(f"  <Play>{audio_url}</Play>")
        elif text:
            parts.append(f"  <Say>{text}</Say>")
        parts.append("</Response>")
        return "\n".join(parts)

    @staticmethod
    def validate_twilio_signature(
        url: str,
        params: dict[str, str],
        signature: str,
        auth_token: str,
    ) -> bool:
        """Validate a Twilio request signature for security.

        Parameters
        ----------
        url:
            The full URL of the webhook endpoint.
        params:
            The POST parameters from Twilio.
        signature:
            The ``X-Twilio-Signature`` header value.
        auth_token:
            Your Twilio auth token.

        Returns
        -------
        bool
            ``True`` if the signature is valid.
        """
        try:
            from twilio.request_validator import RequestValidator
        except ImportError:
            raise ImportError(
                "twilio is required for signature validation. "
                "Install it with: pip install twilio"
            )

        validator = RequestValidator(auth_token)
        return validator.validate(url, params, signature)
