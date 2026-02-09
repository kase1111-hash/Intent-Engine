"""Tests for the REST API server (FastAPI)."""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intent_engine.integrations.server import _suffix_from_filename


# -- Helper function tests --


class TestSuffixFromFilename:
    def test_wav_file(self) -> None:
        assert _suffix_from_filename("recording.wav") == ".wav"

    def test_mp3_file(self) -> None:
        assert _suffix_from_filename("audio.mp3") == ".mp3"

    def test_no_extension(self) -> None:
        assert _suffix_from_filename("audiofile") == ".wav"

    def test_none_filename(self) -> None:
        assert _suffix_from_filename(None) == ".wav"

    def test_empty_string(self) -> None:
        assert _suffix_from_filename("") == ".wav"

    def test_path_with_dots(self) -> None:
        assert _suffix_from_filename("my.audio.file.ogg") == ".ogg"


# -- App creation tests --


class TestCreateApp:
    def test_creates_fastapi_app(self) -> None:
        pytest.importorskip("fastapi")

        from intent_engine.integrations.server import create_app

        mock_engine = MagicMock()
        app = create_app(engine=mock_engine)

        assert app is not None
        assert app.title == "Intent Engine API"
        assert app.version == "0.8.0"

    def test_app_has_required_routes(self) -> None:
        pytest.importorskip("fastapi")

        from intent_engine.integrations.server import create_app

        app = create_app(engine=MagicMock())
        route_paths = {route.path for route in app.routes}

        assert "/process" in route_paths
        assert "/generate" in route_paths
        assert "/synthesize" in route_paths
        assert "/health" in route_paths

    def test_creates_engine_from_kwargs(self) -> None:
        pytest.importorskip("fastapi")

        from intent_engine.integrations.server import create_app

        with patch("intent_engine.engine.IntentEngine") as MockEngine:
            MockEngine.return_value = MagicMock()
            app = create_app(stt_provider="whisper-prosody")
            MockEngine.assert_called_once_with(stt_provider="whisper-prosody")


# -- Endpoint tests with TestClient --


class TestEndpoints:
    @pytest.fixture(autouse=True)
    def _check_deps(self) -> None:
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

    def _make_app(self, engine: MagicMock | None = None):
        from intent_engine.integrations.server import create_app

        return create_app(engine=engine or MagicMock())

    def test_health_endpoint(self) -> None:
        from starlette.testclient import TestClient

        app = self._make_app()
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.8.0"

    def test_generate_endpoint(self) -> None:
        from starlette.testclient import TestClient

        from intent_engine.models.response import Response

        engine = MagicMock()
        engine.generate_response = AsyncMock(
            return_value=Response(text="Hello!", emotion="joyful")
        )

        app = self._make_app(engine)
        client = TestClient(app)

        resp = client.post("/generate", json={"iml": "<utterance>Hi</utterance>"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Hello!"
        assert data["emotion"] == "joyful"

    def test_generate_with_context_and_tone(self) -> None:
        from starlette.testclient import TestClient

        from intent_engine.models.response import Response

        engine = MagicMock()
        engine.generate_response = AsyncMock(
            return_value=Response(text="ok", emotion="neutral")
        )

        app = self._make_app(engine)
        client = TestClient(app)

        resp = client.post(
            "/generate",
            json={
                "iml": "<iml/>",
                "context": "Support call",
                "tone": "empathetic",
            },
        )
        assert resp.status_code == 200
        engine.generate_response.assert_called_once()
        call_kwargs = engine.generate_response.call_args
        assert call_kwargs.kwargs["context"] == "Support call"
        assert call_kwargs.kwargs["tone"] == "empathetic"

    def test_synthesize_endpoint(self) -> None:
        from starlette.testclient import TestClient

        from intent_engine.models.audio import Audio

        audio_data = b"fake audio bytes"
        engine = MagicMock()
        engine.synthesize_speech = AsyncMock(
            return_value=Audio(
                data=audio_data,
                format="wav",
                sample_rate=22050,
                duration=1.5,
            )
        )

        app = self._make_app(engine)
        client = TestClient(app)

        resp = client.post(
            "/synthesize",
            json={"text": "Hello", "emotion": "joyful"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert base64.b64decode(data["audio_data"]) == audio_data
        assert data["format"] == "wav"
        assert data["sample_rate"] == 22050
        assert data["duration"] == 1.5

    def test_synthesize_default_emotion(self) -> None:
        from starlette.testclient import TestClient

        from intent_engine.models.audio import Audio

        engine = MagicMock()
        engine.synthesize_speech = AsyncMock(
            return_value=Audio(data=b"data")
        )

        app = self._make_app(engine)
        client = TestClient(app)

        resp = client.post("/synthesize", json={"text": "Test"})
        assert resp.status_code == 200
        call_kwargs = engine.synthesize_speech.call_args
        assert call_kwargs.kwargs["emotion"] == "neutral"

    def test_generate_error_returns_500(self) -> None:
        from starlette.testclient import TestClient

        engine = MagicMock()
        engine.generate_response = AsyncMock(
            side_effect=RuntimeError("LLM down")
        )

        app = self._make_app(engine)
        client = TestClient(app)

        resp = client.post("/generate", json={"iml": "<iml/>"})
        assert resp.status_code == 500

    def test_synthesize_error_returns_500(self) -> None:
        from starlette.testclient import TestClient

        engine = MagicMock()
        engine.synthesize_speech = AsyncMock(
            side_effect=RuntimeError("TTS down")
        )

        app = self._make_app(engine)
        client = TestClient(app)

        resp = client.post("/synthesize", json={"text": "Hello"})
        assert resp.status_code == 500

    def test_process_endpoint(self) -> None:
        from starlette.testclient import TestClient

        from intent_engine.models.result import Result
        from prosody_protocol import IMLDocument, Segment, Utterance

        engine = MagicMock()
        iml_doc = IMLDocument(
            utterances=(Utterance(children=(Segment(),)),), version="1.0"
        )
        engine.process_voice_input = AsyncMock(
            return_value=Result(
                text="Hello",
                emotion="joyful",
                confidence=0.85,
                iml="<iml><utterance>Hello</utterance></iml>",
                iml_document=iml_doc,
                suggested_tone="joyful",
                prosody_features=[],
            )
        )

        app = self._make_app(engine)
        client = TestClient(app)

        resp = client.post(
            "/process",
            files={"audio": ("test.wav", b"RIFF fake audio", "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Hello"
        assert data["emotion"] == "joyful"
        assert data["confidence"] == 0.85
        assert data["suggested_tone"] == "joyful"
