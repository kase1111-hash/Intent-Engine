"""Twilio voice webhook handler.

Provides ``TwilioVoiceHandler``, a class that processes incoming
Twilio voice webhooks through the Intent Engine pipeline and
returns TwiML responses.

Usage (from the repository root)::

    from intent_engine import IntentEngine
    from examples.integrations.twilio_voice import TwilioVoiceHandler

    engine = IntentEngine()
    handler = TwilioVoiceHandler(engine)

    # In a Flask/FastAPI route, after validating X-Twilio-Signature with
    # TwilioVoiceHandler.validate_twilio_signature():
    twiml = await handler.handle_voice(recording_url, form_data)

By default the reply is spoken with Twilio's own ``<Say>`` voice.  To play
audio synthesized by the engine (with the emotion-mapped TTS voice) instead,
pass an ``audio_publisher``: Twilio can only ``<Play>`` a URL, and the
engine returns raw bytes, so you decide where those bytes are served from.

Security: ``handle_voice`` trusts ``recording_url``.  Only call it for
requests whose signature you have validated; the default downloader also
refuses anything but ``https://*.twilio.com`` and caps the download size.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Iterable
from typing import Any
from xml.sax.saxutils import escape

from intent_engine.models.audio import Audio
from intent_engine.models.result import Result

from ._common import (
    DEFAULT_MAX_DOWNLOAD_BYTES,
    download_media,
    emotion_reported,
    temp_audio_file,
)

logger = logging.getLogger(__name__)

TWILIO_HOSTS = ("twilio.com",)
"""Hosts (and their subdomains) recordings are downloaded from by default."""

_APOLOGY = "I'm sorry, I'm having trouble processing your request. Please try again."

# Characters XML 1.0 cannot represent at all, even escaped.
_INVALID_XML_CHARS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


def _xml_text(value: str) -> str:
    """Escape *value* for use as XML character data."""
    return escape(_INVALID_XML_CHARS.sub("", value))


class TwilioVoiceHandler:
    """Process Twilio voice webhooks through the Intent Engine pipeline.

    Receives a recording URL from Twilio, downloads the audio, runs it
    through ``IntentEngine.process_voice_input()``, picks a reply (from
    ``response_callback``, or a canned one based on the reported emotion)
    and returns it as TwiML.  It does not call the LLM; have
    ``response_callback`` do that if you want generated replies.

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
        audio from Twilio.  Defaults to a size-capped ``httpx`` download
        restricted to ``https://*.twilio.com``; a custom function is
        responsible for its own restrictions.
    audio_publisher:
        Optional async callable ``(Audio) -> str`` that makes synthesized
        audio available to Twilio and returns the URL to ``<Play>``.  When
        given, the reply is synthesized with the engine's TTS; if
        synthesis or publishing fails the reply is spoken with ``<Say>``
        instead.  When omitted, TTS is not called at all.
    max_download_bytes:
        Largest recording the default downloader accepts.
    allowed_hosts:
        Hosts (and their subdomains) the default downloader may fetch from.
    reply_emotion:
        Emotion the reply is synthesized with (only used with
        ``audio_publisher``).  The caller's own emotion is never used:
        ``Result.suggested_tone`` describes the caller, and the reply here is
        not an LLM ``Response``, so it has no ``Response.emotion`` of its own.
        An angry caller's de-escalation reply should not be spoken angrily.
    """

    def __init__(
        self,
        engine: Any,
        response_callback: Callable[[Result], str] | None = None,
        download_func: Callable[..., Any] | None = None,
        *,
        audio_publisher: Callable[[Audio], Awaitable[str]] | None = None,
        max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
        allowed_hosts: Iterable[str] = TWILIO_HOSTS,
        reply_emotion: str = "neutral",
    ) -> None:
        self._engine = engine
        self._response_callback = response_callback
        self._download_func = download_func
        self._audio_publisher = audio_publisher
        self._max_download_bytes = max_download_bytes
        self._allowed_hosts = tuple(allowed_hosts)
        self._reply_emotion = reply_emotion

    async def _download_audio(self, url: str) -> bytes:
        """Download audio bytes from a URL."""
        if self._download_func:
            result: bytes = await self._download_func(url)
            return result

        return await download_media(
            url,
            allowed_hosts=self._allowed_hosts,
            max_bytes=self._max_download_bytes,
        )

    async def handle_voice(
        self,
        recording_url: str,
        form_data: dict[str, str] | None = None,
    ) -> str:
        """Process a Twilio voice webhook and return TwiML XML.

        Never raises for a bad or undecodable recording or a pipeline
        failure: the caller hears a short apology instead, and the details
        are logged.

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

        try:
            audio_bytes = await self._download_audio(recording_url)

            async with temp_audio_file(audio_bytes) as tmp_path:
                result = await self._engine.process_voice_input(tmp_path)

            if self._response_callback:
                response_text = self._response_callback(result)
            else:
                response_text = self._default_response(result)

            return await self._respond(response_text)

        except Exception:
            # A failed recording must not turn into Twilio's "application
            # error" message; details stay in the log, not on the call.
            logger.exception("Error processing Twilio call %s", call_sid)
            return self._build_twiml(text=_APOLOGY)

    async def _respond(self, text: str) -> str:
        """Build the TwiML for *text*, using synthesized audio when possible."""
        if self._audio_publisher is None:
            return self._build_twiml(text=text)

        try:
            audio = await self._engine.synthesize_speech(text, emotion=self._reply_emotion)
            audio_url = await self._audio_publisher(audio)
        except Exception:
            logger.exception("Could not synthesize or publish the reply; using <Say>")
            return self._build_twiml(text=text)
        return self._build_twiml(audio_url=audio_url, text=text)

    @staticmethod
    def _default_response(result: Result) -> str:
        """Generate a default response based on emotion detection.

        The emotion branches only apply when the engine reported an
        emotion; ``("neutral", 0.0)`` means it did not.
        """
        if emotion_reported(result):
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
        falls back to ``<Say>``.  Text and URLs are XML-escaped, so a
        transcript such as ``AT&T`` or one containing markup cannot break
        the document or add verbs to it.
        """
        parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<Response>"]
        if audio_url:
            parts.append(f"  <Play>{_xml_text(audio_url)}</Play>")
        elif text:
            parts.append(f"  <Say>{_xml_text(text)}</Say>")
        parts.append("</Response>")
        return "\n".join(parts)

    @staticmethod
    def validate_twilio_signature(
        url: str,
        params: dict[str, str],
        signature: str | None,
        auth_token: str,
    ) -> bool:
        """Validate a Twilio request signature for security.

        Call this on every webhook request before ``handle_voice``.

        Parameters
        ----------
        url:
            The full URL of the webhook endpoint.
        params:
            The POST parameters from Twilio.
        signature:
            The ``X-Twilio-Signature`` header value; ``None`` (the header
            is absent) never validates.
        auth_token:
            Your Twilio auth token.

        Returns
        -------
        bool
            ``True`` if the signature is valid; ``False`` for a wrong,
            missing or malformed signature or a malformed *url*.

        Raises
        ------
        ValueError
            If *auth_token* is empty.  A signature made with an empty key
            can be computed by anyone, so an unset ``TWILIO_AUTH_TOKEN``
            must be a loud configuration error, not a check that passes.
        """
        if not isinstance(auth_token, str) or not auth_token.strip():
            raise ValueError("auth_token must be a non-empty string (is TWILIO_AUTH_TOKEN set?)")

        try:
            from twilio.request_validator import RequestValidator
        except ImportError as exc:
            raise ImportError(
                "twilio is required for signature validation. "
                "Install it with: pip install twilio"
            ) from exc

        validator = RequestValidator(auth_token)
        try:
            return bool(validator.validate(url, params, signature))
        except (TypeError, ValueError):
            # No signature header (None), or a malformed url such as a bad port or
            # IPv6 host: not a valid request.  The token is checked above, outside
            # the try, so a misconfiguration is never reported as a bad request.
            return False
