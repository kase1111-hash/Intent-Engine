"""CloudEngine -- managed cloud deployment.

Wraps a REST client that calls the Intent Engine managed service.
Handles authentication, retries with exponential backoff, and
rate limiting.  Returns the same ``Result``, ``Response``, and
``Audio`` objects as the base ``IntentEngine``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from intent_engine.errors import (
    IntentEngineError,
    LLMError,
    STTError,
    TTSError,
)
from intent_engine.models.audio import Audio
from intent_engine.models.response import Response
from intent_engine.models.result import Result

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.intentengine.ai/v1"
_DEFAULT_TIMEOUT = 30.0
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


class CloudEngine:
    """Managed cloud deployment of Intent Engine.

    All processing is handled by Intent Engine infrastructure.
    Audio is encrypted in transit (TLS 1.3), used only for
    temporary processing, and deleted after the response.

    Parameters
    ----------
    api_key:
        Intent Engine API key (``ie_sk_...``).
    base_url:
        Base URL for the managed service.  Defaults to the
        production Intent Engine API.
    timeout:
        HTTP request timeout in seconds.
    max_retries:
        Maximum number of retry attempts for transient failures.
    zero_logging:
        If ``True``, request zero-logging mode (no request data
        is stored server-side).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _MAX_RETRIES,
        zero_logging: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for CloudEngine")

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._zero_logging = zero_logging
        self._session: Any = None

        logger.info(
            "CloudEngine initialized (base_url=%s, zero_logging=%s)",
            self._base_url,
            self._zero_logging,
        )

    def _headers(self) -> dict[str, str]:
        """Build default request headers."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "intent-engine-python/0.8.0",
        }
        if self._zero_logging:
            headers["X-Zero-Logging"] = "true"
        return headers

    async def _get_session(self) -> Any:
        """Lazy-initialize the HTTP session."""
        if self._session is None:
            try:
                import httpx  # noqa: F811

                self._session = httpx.AsyncClient(
                    base_url=self._base_url,
                    headers=self._headers(),
                    timeout=self._timeout,
                )
            except ImportError:
                raise ImportError(
                    "httpx is required for CloudEngine. "
                    "Install it with: pip install httpx"
                )
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request with retry logic.

        Retries transient failures (5xx, timeouts, connection errors)
        up to ``max_retries`` times with exponential backoff.
        """
        session = await self._get_session()
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                if files:
                    resp = await session.post(
                        path,
                        files=files,
                        headers={
                            k: v
                            for k, v in self._headers().items()
                            if k != "Content-Type"
                        },
                    )
                elif method == "POST":
                    resp = await session.post(path, json=json_data)
                else:
                    resp = await session.get(path)

                if resp.status_code == 429:
                    retry_after = float(
                        resp.headers.get("Retry-After", _BACKOFF_BASE ** attempt)
                    )
                    logger.warning(
                        "Rate limited; retrying after %.1fs", retry_after
                    )
                    await asyncio.sleep(retry_after)
                    continue

                if resp.status_code >= 500:
                    last_error = IntentEngineError(
                        f"Server error {resp.status_code}: {resp.text}"
                    )
                    if attempt < self._max_retries:
                        wait = _BACKOFF_BASE ** attempt
                        logger.warning(
                            "Server error %d; retrying in %.1fs",
                            resp.status_code,
                            wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    raise last_error

                if resp.status_code == 401:
                    raise IntentEngineError("Invalid API key")
                if resp.status_code == 403:
                    raise IntentEngineError("Forbidden: check API key permissions")
                if resp.status_code >= 400:
                    raise IntentEngineError(
                        f"API error {resp.status_code}: {resp.text}"
                    )

                return resp.json()

            except IntentEngineError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries:
                    wait = _BACKOFF_BASE ** attempt
                    logger.warning(
                        "Request failed (%s); retrying in %.1fs", exc, wait
                    )
                    await asyncio.sleep(wait)
                    continue

        raise IntentEngineError(
            f"Request to {path} failed after {self._max_retries + 1} attempts: "
            f"{last_error}"
        )

    async def process_voice_input(self, audio_path: str) -> Result:
        """Upload audio and process through the cloud pipeline.

        Parameters
        ----------
        audio_path:
            Path to the audio file on disk.

        Returns
        -------
        Result
            Pipeline result with transcription, emotion, and IML.

        Raises
        ------
        STTError
            If transcription fails.
        IntentEngineError
            On network or service errors.
        """
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            data = await self._request(
                "POST",
                "/process",
                files={"audio": (audio_path, audio_bytes, "audio/wav")},
            )

            from prosody_protocol import IMLDocument, IMLParser, SpanFeatures

            for field in ("text", "emotion", "confidence", "iml"):
                if field not in data:
                    raise IntentEngineError(
                        f"Cloud response missing required field: {field!r}"
                    )

            parser = IMLParser()
            iml_doc = parser.parse(data["iml"])

            features = [
                SpanFeatures(
                    start_ms=feat.get("start_ms", 0),
                    end_ms=feat.get("end_ms", 0),
                    text=feat.get("text", ""),
                    f0_mean=feat.get("f0_mean"),
                    speech_rate=feat.get("speech_rate"),
                )
                for feat in data.get("prosody_features", [])
            ]

            return Result(
                text=data["text"],
                emotion=data["emotion"],
                confidence=data["confidence"],
                iml=data["iml"],
                iml_document=iml_doc,
                suggested_tone=data.get("suggested_tone", data["emotion"]),
                prosody_features=features,
            )
        except IntentEngineError:
            raise
        except Exception as exc:
            raise STTError(f"Cloud processing failed: {exc}") from exc

    async def generate_response(
        self,
        iml: str,
        context: str | None = None,
        tone: str | None = None,
    ) -> Response:
        """Generate a response via the cloud LLM service.

        Parameters
        ----------
        iml:
            Serialized IML markup string.
        context:
            Optional conversation context.
        tone:
            Suggested response tone.

        Returns
        -------
        Response
            Generated text and emotion.

        Raises
        ------
        LLMError
            If the LLM service fails.
        """
        payload: dict[str, Any] = {"iml": iml}
        if context:
            payload["context"] = context
        if tone:
            payload["tone"] = tone

        try:
            data = await self._request("POST", "/generate", json_data=payload)
            if "text" not in data:
                raise IntentEngineError(
                    "Cloud generate response missing required field: 'text'"
                )
            return Response(
                text=data["text"],
                emotion=data.get("emotion", "neutral"),
            )
        except IntentEngineError:
            raise
        except Exception as exc:
            raise LLMError(f"Cloud generation failed: {exc}") from exc

    async def synthesize_speech(
        self, text: str, emotion: str = "neutral"
    ) -> Audio:
        """Synthesize speech via the cloud TTS service.

        Parameters
        ----------
        text:
            Text to synthesize.
        emotion:
            Emotion label for voice parameter adjustment.

        Returns
        -------
        Audio
            Synthesized audio bytes and metadata.

        Raises
        ------
        TTSError
            If synthesis fails.
        """
        try:
            data = await self._request(
                "POST",
                "/synthesize",
                json_data={"text": text, "emotion": emotion},
            )

            import base64

            if "audio_data" not in data:
                raise IntentEngineError(
                    "Cloud synthesize response missing required field: 'audio_data'"
                )
            audio_data = base64.b64decode(data["audio_data"])

            return Audio(
                data=audio_data,
                format=data.get("format", "wav"),
                sample_rate=data.get("sample_rate", 22050),
                duration=data.get("duration"),
                url=data.get("url"),
            )
        except IntentEngineError:
            raise
        except Exception as exc:
            raise TTSError(f"Cloud synthesis failed: {exc}") from exc

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session is not None:
            await self._session.aclose()
            self._session = None

    # -- Synchronous convenience wrappers --

    def process_voice_input_sync(self, audio_path: str) -> Result:
        """Synchronous wrapper for :meth:`process_voice_input`."""
        return asyncio.run(self.process_voice_input(audio_path))

    def generate_response_sync(
        self,
        iml: str,
        context: str | None = None,
        tone: str | None = None,
    ) -> Response:
        """Synchronous wrapper for :meth:`generate_response`."""
        return asyncio.run(
            self.generate_response(iml, context=context, tone=tone)
        )

    def synthesize_speech_sync(
        self, text: str, emotion: str = "neutral"
    ) -> Audio:
        """Synchronous wrapper for :meth:`synthesize_speech`."""
        return asyncio.run(self.synthesize_speech(text, emotion=emotion))
