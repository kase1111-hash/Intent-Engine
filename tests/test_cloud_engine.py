"""Tests for CloudEngine (managed cloud deployment)."""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intent_engine.cloud_engine import CloudEngine, _BACKOFF_BASE, _DEFAULT_BASE_URL
from intent_engine.errors import IntentEngineError, LLMError, STTError, TTSError
from intent_engine.models.audio import Audio
from intent_engine.models.response import Response
from intent_engine.models.result import Result


# -- Construction --


class TestCloudEngineConstruction:
    def test_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="api_key is required"):
            CloudEngine(api_key="")

    def test_creates_with_api_key(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test123")
        assert engine._api_key == "ie_sk_test123"

    def test_default_base_url(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test")
        assert engine._base_url == _DEFAULT_BASE_URL

    def test_custom_base_url(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test", base_url="https://custom.api.com/v2/")
        assert engine._base_url == "https://custom.api.com/v2"  # trailing slash stripped

    def test_default_timeout(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test")
        assert engine._timeout == 30.0

    def test_custom_timeout(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test", timeout=60.0)
        assert engine._timeout == 60.0

    def test_zero_logging(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test", zero_logging=True)
        assert engine._zero_logging is True

    def test_session_not_initialized(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test")
        assert engine._session is None


# -- Headers --


class TestHeaders:
    def test_authorization_header(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test")
        headers = engine._headers()
        assert headers["Authorization"] == "Bearer ie_sk_test"

    def test_content_type(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test")
        headers = engine._headers()
        assert headers["Content-Type"] == "application/json"

    def test_zero_logging_header_absent_by_default(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test")
        headers = engine._headers()
        assert "X-Zero-Logging" not in headers

    def test_zero_logging_header_present(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test", zero_logging=True)
        headers = engine._headers()
        assert headers["X-Zero-Logging"] == "true"


# -- Request with retries --


class TestRequestRetries:
    def _make_engine(self, max_retries: int = 2) -> CloudEngine:
        return CloudEngine(api_key="ie_sk_test", max_retries=max_retries)

    def test_401_raises_immediately(self) -> None:
        engine = self._make_engine()
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_session.post = AsyncMock(return_value=mock_resp)
        engine._session = mock_session

        with pytest.raises(IntentEngineError, match="Invalid API key"):
            asyncio.run(
                engine._request("POST", "/test", json_data={})
            )

    def test_403_raises_immediately(self) -> None:
        engine = self._make_engine()
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        mock_session.post = AsyncMock(return_value=mock_resp)
        engine._session = mock_session

        with pytest.raises(IntentEngineError, match="Forbidden"):
            asyncio.run(
                engine._request("POST", "/test", json_data={})
            )

    def test_200_returns_json(self) -> None:
        engine = self._make_engine()
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True}
        mock_session.post = AsyncMock(return_value=mock_resp)
        engine._session = mock_session

        result = asyncio.run(
            engine._request("POST", "/test", json_data={})
        )
        assert result == {"ok": True}

    @patch("intent_engine.cloud_engine.asyncio.sleep", new_callable=AsyncMock)
    def test_500_retries(self, mock_sleep: AsyncMock) -> None:
        engine = self._make_engine(max_retries=2)
        mock_session = MagicMock()

        # First two calls: 500, third call: 200
        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.text = "Internal Server Error"

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"recovered": True}

        mock_session.post = AsyncMock(side_effect=[resp_500, resp_500, resp_200])
        engine._session = mock_session

        result = asyncio.run(
            engine._request("POST", "/test", json_data={})
        )
        assert result == {"recovered": True}
        assert mock_session.post.call_count == 3

    @patch("intent_engine.cloud_engine.asyncio.sleep", new_callable=AsyncMock)
    def test_500_exhausts_retries(self, mock_sleep: AsyncMock) -> None:
        engine = self._make_engine(max_retries=1)
        mock_session = MagicMock()

        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.text = "Server Error"

        mock_session.post = AsyncMock(return_value=resp_500)
        engine._session = mock_session

        with pytest.raises(IntentEngineError, match="Server error 500"):
            asyncio.run(
                engine._request("POST", "/test", json_data={})
            )


# -- process_voice_input --


class TestCloudProcessVoiceInput:
    def _make_engine_with_mock_request(
        self, response_data: dict,
    ) -> CloudEngine:
        engine = CloudEngine(api_key="ie_sk_test")
        engine._request = AsyncMock(return_value=response_data)
        return engine

    def test_returns_result(self) -> None:
        data = {
            "text": "Hello world",
            "emotion": "joyful",
            "confidence": 0.85,
            "iml": "<iml><utterance>Hello world</utterance></iml>",
            "suggested_tone": "joyful",
            "prosody_features": [],
        }
        engine = self._make_engine_with_mock_request(data)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake audio")
            audio_path = f.name

        try:
            result = asyncio.run(
                engine.process_voice_input(audio_path)
            )
            assert isinstance(result, Result)
            assert result.text == "Hello world"
            assert result.emotion == "joyful"
            assert result.confidence == 0.85
        finally:
            import os
            os.unlink(audio_path)

    def test_stt_error_on_failure(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test")
        engine._request = AsyncMock(side_effect=RuntimeError("network down"))

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF fake")
            audio_path = f.name

        try:
            with pytest.raises(STTError, match="Cloud processing failed"):
                asyncio.run(
                    engine.process_voice_input(audio_path)
                )
        finally:
            import os
            os.unlink(audio_path)


# -- generate_response --


class TestCloudGenerateResponse:
    def test_returns_response(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test")
        engine._request = AsyncMock(return_value={
            "text": "Hello! How can I help?",
            "emotion": "joyful",
        })

        result = asyncio.run(
            engine.generate_response("<utterance>Hello</utterance>")
        )
        assert isinstance(result, Response)
        assert result.text == "Hello! How can I help?"
        assert result.emotion == "joyful"

    def test_passes_context_and_tone(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test")
        engine._request = AsyncMock(return_value={
            "text": "ok", "emotion": "neutral"
        })

        asyncio.run(
            engine.generate_response("<iml/>", context="Support", tone="empathetic")
        )

        call_kwargs = engine._request.call_args
        payload = call_kwargs.kwargs["json_data"]
        assert payload["context"] == "Support"
        assert payload["tone"] == "empathetic"

    def test_llm_error_on_failure(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test")
        engine._request = AsyncMock(side_effect=RuntimeError("API error"))

        with pytest.raises(LLMError, match="Cloud generation failed"):
            asyncio.run(
                engine.generate_response("<iml/>")
            )


# -- synthesize_speech --


class TestCloudSynthesizeSpeech:
    def test_returns_audio(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test")
        audio_bytes = b"fake audio data"
        engine._request = AsyncMock(return_value={
            "audio_data": base64.b64encode(audio_bytes).decode(),
            "format": "wav",
            "sample_rate": 22050,
            "duration": 1.5,
        })

        result = asyncio.run(
            engine.synthesize_speech("Hello", emotion="joyful")
        )
        assert isinstance(result, Audio)
        assert result.data == audio_bytes
        assert result.format == "wav"
        assert result.sample_rate == 22050
        assert result.duration == 1.5

    def test_passes_emotion(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test")
        engine._request = AsyncMock(return_value={
            "audio_data": base64.b64encode(b"data").decode(),
        })

        asyncio.run(
            engine.synthesize_speech("Test", emotion="sad")
        )

        call_kwargs = engine._request.call_args
        payload = call_kwargs.kwargs["json_data"]
        assert payload["emotion"] == "sad"

    def test_tts_error_on_failure(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test")
        engine._request = AsyncMock(side_effect=RuntimeError("TTS down"))

        with pytest.raises(TTSError, match="Cloud synthesis failed"):
            asyncio.run(
                engine.synthesize_speech("Hello")
            )


# -- close --


class TestCloudClose:
    def test_close_session(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test")
        mock_session = MagicMock()
        mock_session.aclose = AsyncMock()
        engine._session = mock_session

        asyncio.run(engine.close())
        mock_session.aclose.assert_called_once()
        assert engine._session is None

    def test_close_no_session(self) -> None:
        engine = CloudEngine(api_key="ie_sk_test")
        # Should not raise
        asyncio.run(engine.close())
