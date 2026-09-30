"""Tests for the REST API server (FastAPI)."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import importlib.metadata
import logging
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prosody_protocol import AudioProcessingError, IMLParseError, IMLValidator, SpanFeatures

from examples.integrations.rest_server import create_app, create_app_from_env
from intent_engine.engine import IntentEngine
from intent_engine.errors import IntentEngineError, LLMError, STTError, TTSError
from intent_engine.models.audio import Audio
from intent_engine.models.response import Response
from intent_engine.models.result import Result
from intent_engine.stt.base import STTProvider, TranscriptionResult

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

MakeResult = Callable[..., Result]

VALID_IML = "<utterance>Hi</utterance>"
PACKAGE_VERSION = importlib.metadata.version("intent-engine")


def _engine(result: Result | None = None) -> MagicMock:
    engine = MagicMock()
    engine.process_voice_input = AsyncMock(return_value=result)
    engine.generate_response = AsyncMock(return_value=Response(text="Hello!", emotion="joyful"))
    engine.synthesize_speech = AsyncMock(return_value=Audio(data=b"audio", format="wav"))
    return engine


def _test_client(app: Any) -> Any:
    from starlette.testclient import TestClient

    # Errors must be answered by the app itself, not re-raised into the test.
    return TestClient(app, raise_server_exceptions=False)


def _client(engine: MagicMock | None = None, **kwargs: Any) -> Any:
    return _test_client(create_app(engine=engine or _engine(), **kwargs))


def _upload(data: bytes, filename: str = "test.wav", content_type: str = "audio/wav") -> Any:
    return {"audio": (filename, data, content_type)}


# -- App creation tests --


class TestCreateApp:
    def test_creates_fastapi_app(self) -> None:
        app = create_app(engine=MagicMock())

        assert app is not None
        assert app.title == "Intent Engine API"
        assert app.version == PACKAGE_VERSION

    def test_app_has_required_routes(self) -> None:
        app = create_app(engine=MagicMock())
        route_paths = {route.path for route in app.routes}

        assert "/process" in route_paths
        assert "/generate" in route_paths
        assert "/synthesize" in route_paths
        assert "/health" in route_paths

    def test_creates_engine_from_kwargs(self) -> None:
        with patch("intent_engine.engine.IntentEngine") as MockEngine:
            MockEngine.return_value = MagicMock()
            create_app(stt_provider="whisper-prosody")
            MockEngine.assert_called_once_with(stt_provider="whisper-prosody")

    def test_apps_do_not_share_state(self) -> None:
        first, second = _engine(), _engine()
        client_a, client_b = _client(first), _client(second)

        client_a.post("/synthesize", json={"text": "a"})

        first.synthesize_speech.assert_called_once()
        second.synthesize_speech.assert_not_called()
        assert client_b.get("/health").status_code == 200

    def test_empty_api_key_is_a_configuration_error(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            create_app(engine=MagicMock(), api_key="")

    def test_missing_multipart_gives_an_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "python_multipart", None)
        monkeypatch.setitem(sys.modules, "multipart", None)

        with pytest.raises(ImportError, match="python-multipart"):
            create_app(engine=MagicMock())


# -- The documented way to run the server: uvicorn --factory --


class TestFactory:
    @pytest.fixture()
    def env_engine(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
        for name in (
            "INTENT_STT_PROVIDER",
            "INTENT_LLM_PROVIDER",
            "INTENT_TTS_PROVIDER",
            "INTENT_API_KEY",
        ):
            monkeypatch.delenv(name, raising=False)
        with patch("intent_engine.engine.IntentEngine") as MockEngine:
            MockEngine.return_value = _engine()
            yield MockEngine

    def test_builds_a_working_app(self, env_engine: MagicMock) -> None:
        client = _test_client(create_app_from_env())

        assert client.get("/health").status_code == 200
        env_engine.assert_called_once_with()

    def test_providers_come_from_the_environment(
        self, env_engine: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INTENT_STT_PROVIDER", "deepgram")
        monkeypatch.setenv("INTENT_LLM_PROVIDER", "openai")
        monkeypatch.setenv("INTENT_TTS_PROVIDER", "espeak")

        create_app_from_env()

        env_engine.assert_called_once_with(
            stt_provider="deepgram", llm_provider="openai", tts_provider="espeak"
        )

    def test_api_key_comes_from_the_environment(
        self, env_engine: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INTENT_API_KEY", "s3cret")
        client = _test_client(create_app_from_env())

        assert client.post("/synthesize", json={"text": "hi"}).status_code == 401
        response = client.post(
            "/synthesize", json={"text": "hi"}, headers={"X-API-Key": "s3cret"}
        )
        assert response.status_code == 200

    def test_warns_when_running_without_an_api_key(
        self, env_engine: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            create_app_from_env()

        assert "INTENT_API_KEY" in caplog.text

    @pytest.mark.parametrize("value", ["", "   "])
    def test_an_empty_api_key_is_refused_not_treated_as_no_authentication(
        self, env_engine: MagicMock, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        # `INTENT_API_KEY=${INTENT_API_KEY}` in a compose file or k8s template
        # expands to "" when the variable is undefined; that must not start an
        # open server.  Only a variable that is not set at all means "no key".
        monkeypatch.setenv("INTENT_API_KEY", value)

        with pytest.raises(ValueError, match="INTENT_API_KEY"):
            create_app_from_env()

        env_engine.assert_not_called()  # refused before any provider is built

    def test_an_unset_api_key_is_still_allowed(self, env_engine: MagicMock) -> None:
        client = _test_client(create_app_from_env())

        assert client.post("/synthesize", json={"text": "hi"}).status_code == 200

    def test_empty_provider_variables_keep_the_defaults(
        self, env_engine: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("INTENT_STT_PROVIDER", "INTENT_LLM_PROVIDER", "INTENT_TTS_PROVIDER"):
            monkeypatch.setenv(name, "")

        create_app_from_env()

        env_engine.assert_called_once_with()

    def test_uvicorn_serves_the_factory(self, env_engine: MagicMock) -> None:
        # The command the README documents:
        #   uvicorn --factory examples.integrations.rest_server:create_app_from_env
        uvicorn = pytest.importorskip("uvicorn")
        import httpx

        config = uvicorn.Config(
            "examples.integrations.rest_server:create_app_from_env",
            factory=True,
            host="127.0.0.1",
            port=0,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 15
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert server.started, "uvicorn did not start the factory app"
            port = server.servers[0].sockets[0].getsockname()[1]

            response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=10)

            assert response.status_code == 200
            assert response.json()["status"] == "ok"
        finally:
            server.should_exit = True
            thread.join(timeout=15)


# -- Endpoint tests --


class TestEndpoints:
    def test_health_endpoint(self) -> None:
        resp = _client().get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == PACKAGE_VERSION

    def test_generate_endpoint(self) -> None:
        engine = _engine()
        resp = _client(engine).post("/generate", json={"iml": VALID_IML})

        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Hello!"
        assert data["emotion"] == "joyful"

    def test_generate_with_context_and_tone(self) -> None:
        engine = _engine()
        engine.generate_response = AsyncMock(return_value=Response(text="ok", emotion="neutral"))

        resp = _client(engine).post(
            "/generate",
            json={"iml": VALID_IML, "context": "Support call", "tone": "empathetic"},
        )

        assert resp.status_code == 200
        engine.generate_response.assert_called_once()
        call_kwargs = engine.generate_response.call_args
        assert call_kwargs.kwargs["context"] == "Support call"
        assert call_kwargs.kwargs["tone"] == "empathetic"

    def test_synthesize_endpoint(self) -> None:
        audio_data = b"fake audio bytes"
        engine = _engine()
        engine.synthesize_speech = AsyncMock(
            return_value=Audio(data=audio_data, format="wav", sample_rate=22050, duration=1.5)
        )

        resp = _client(engine).post("/synthesize", json={"text": "Hello", "emotion": "joyful"})

        assert resp.status_code == 200
        data = resp.json()
        assert base64.b64decode(data["audio_data"]) == audio_data
        assert data["format"] == "wav"
        assert data["sample_rate"] == 22050
        assert data["duration"] == 1.5

    def test_synthesize_default_emotion(self) -> None:
        engine = _engine()

        resp = _client(engine).post("/synthesize", json={"text": "Test"})

        assert resp.status_code == 200
        assert engine.synthesize_speech.call_args.kwargs["emotion"] == "neutral"

    def test_process_endpoint(self, make_result: MakeResult, wav_bytes: bytes) -> None:
        result = make_result(text="Hello", emotion="joyful", confidence=0.85)
        engine = _engine(result)

        resp = _client(engine).post("/process", files=_upload(wav_bytes))

        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Hello"
        assert data["emotion"] == "joyful"
        assert data["confidence"] == 0.85
        assert data["suggested_tone"] == "joyful"

    def test_process_passes_abstention_through(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        # ("neutral", 0.0) is the engine's "no emotion reported"; do not invent one.
        engine = _engine(make_result(text="Hello"))

        data = _client(engine).post("/process", files=_upload(wav_bytes)).json()

        assert (data["emotion"], data["confidence"]) == ("neutral", 0.0)

    def test_process_serializes_prosody_features(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        feature = SpanFeatures(start_ms=0, end_ms=400, text="Hello", f0_mean=180.5, speech_rate=4.2)
        result = dataclasses.replace(make_result(text="Hello"), prosody_features=[feature])

        data = _client(_engine(result)).post("/process", files=_upload(wav_bytes)).json()

        assert data["prosody_features"] == [
            {"start_ms": 0, "end_ms": 400, "text": "Hello", "f0_mean": 180.5, "speech_rate": 4.2}
        ]

    def test_process_hands_the_engine_the_uploaded_bytes(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        seen: dict[str, Any] = {}

        async def process(path: str) -> Result:
            seen["suffix"] = Path(path).suffix
            seen["data"] = Path(path).read_bytes()
            seen["path"] = path
            return make_result()

        engine = _engine()
        engine.process_voice_input = process

        _client(engine).post("/process", files=_upload(wav_bytes))

        assert seen["data"] == wav_bytes
        assert seen["suffix"] == ".wav"
        assert not Path(seen["path"]).exists()  # cleaned up after the request


# -- Upload validation --


class TestUploadValidation:
    def test_non_audio_upload_is_415(self) -> None:
        engine = _engine()

        resp = _client(engine).post(
            "/process", files=_upload(b"just some text", "notes.txt", "text/plain")
        )

        assert resp.status_code == 415
        engine.process_voice_input.assert_not_called()

    def test_content_decides_not_the_file_name(self) -> None:
        engine = _engine()

        resp = _client(engine).post(
            "/process", files=_upload(b"<html>login</html>", "clip.wav", "audio/wav")
        )

        assert resp.status_code == 415
        engine.process_voice_input.assert_not_called()

    def test_audio_with_a_misleading_name_is_accepted(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        engine = _engine(make_result())

        resp = _client(engine).post("/process", files=_upload(wav_bytes, "clip.txt", "text/plain"))

        assert resp.status_code == 200

    def test_empty_upload_is_422(self) -> None:
        engine = _engine()

        resp = _client(engine).post("/process", files=_upload(b""))

        assert resp.status_code == 422
        engine.process_voice_input.assert_not_called()

    def test_very_long_file_extension_is_harmless(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        engine = _engine(make_result())

        resp = _client(engine).post("/process", files=_upload(wav_bytes, "a." + "x" * 300))

        assert resp.status_code == 200

    def test_missing_audio_field_is_422(self) -> None:
        assert _client().post("/process", files={"other": ("a.wav", b"RIFF")}).status_code == 422


class TestSizeLimits:
    def test_oversize_upload_is_413(self, wav_bytes: bytes) -> None:
        engine = _engine()
        client = _client(engine, max_upload_bytes=1000)

        resp = client.post("/process", files=_upload(wav_bytes + b"\x00" * 5000))

        assert resp.status_code == 413
        engine.process_voice_input.assert_not_called()

    def test_upload_just_over_the_limit_is_413(self, wav_bytes: bytes) -> None:
        # Small enough to pass the request-size check (which allows for the
        # multipart framing) but larger than the audio limit itself.
        engine = _engine()
        client = _client(engine, max_upload_bytes=len(wav_bytes) - 1)

        resp = client.post("/process", files=_upload(wav_bytes))

        assert resp.status_code == 413
        engine.process_voice_input.assert_not_called()

    def test_upload_at_the_limit_is_accepted(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        engine = _engine(make_result())
        client = _client(engine, max_upload_bytes=len(wav_bytes))

        assert client.post("/process", files=_upload(wav_bytes)).status_code == 200

    def test_chunked_upload_without_content_length_is_413(self, wav_bytes: bytes) -> None:
        engine = _engine()
        client = _client(engine, max_upload_bytes=1000)
        boundary = "xBOUNDARYx"
        head = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="audio"; filename="a.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        tail = f"\r\n--{boundary}--\r\n".encode()

        def body() -> Iterator[bytes]:
            yield head
            yield wav_bytes
            for _ in range(200):
                yield b"\x00" * 1024
            yield tail

        resp = client.post(
            "/process",
            content=body(),
            headers={"content-type": f"multipart/form-data; boundary={boundary}"},
        )

        assert resp.status_code == 413
        engine.process_voice_input.assert_not_called()

    def test_oversize_json_body_is_413(self) -> None:
        engine = _engine()

        resp = _client(engine).post("/generate", json={"iml": "<utterance>" + "a" * 5_000_000})

        assert resp.status_code == 413
        engine.generate_response.assert_not_called()

    @pytest.mark.parametrize("path", ["/generate", "/synthesize"])
    def test_json_endpoints_are_limited_whatever_content_type_the_client_claims(
        self, path: str
    ) -> None:
        # The upload limit is for /process only.  Choosing the limit from the
        # request's Content-Type lets any client get the 25 MiB one on the
        # JSON endpoints by calling its body multipart/form-data.
        engine = _engine()

        resp = _client(engine).post(
            path,
            content=b"a" * 5_000_000,
            headers={"content-type": "multipart/form-data; boundary=zz"},
        )

        assert resp.status_code == 413
        assert "payload_too_large" in resp.text
        engine.generate_response.assert_not_called()
        engine.synthesize_speech.assert_not_called()

    @staticmethod
    def _small_limits(engine: MagicMock) -> Any:
        # The middleware allows 64 KiB of multipart framing on top of the upload
        # limit, so a 200 KB request is stopped by the middleware, before Starlette
        # spools it, and not by the handler's own byte count afterwards.
        return _client(engine, max_upload_bytes=1000)

    def test_the_size_middleware_rejects_a_declared_oversize_upload(
        self, wav_bytes: bytes
    ) -> None:
        engine = _engine()

        resp = self._small_limits(engine).post(
            "/process", files=_upload(wav_bytes + b"\x00" * 200_000)
        )

        assert resp.status_code == 413
        assert "payload_too_large" in resp.text  # answered by the middleware, not the handler
        engine.process_voice_input.assert_not_called()

    def test_the_size_middleware_rejects_a_chunked_oversize_upload(self, wav_bytes: bytes) -> None:
        engine = _engine()
        head = (
            b'--x\r\nContent-Disposition: form-data; name="audio"; filename="a.wav"\r\n'
            b"Content-Type: audio/wav\r\n\r\n"
        )

        def body() -> Iterator[bytes]:
            yield head
            yield wav_bytes
            for _ in range(200):
                yield b"\x00" * 1024
            yield b"\r\n--x--\r\n"

        resp = self._small_limits(engine).post(
            "/process", content=body(), headers={"content-type": "multipart/form-data; boundary=x"}
        )

        assert resp.status_code == 413
        assert "payload_too_large" in resp.text
        engine.process_voice_input.assert_not_called()

    def test_an_upload_larger_than_the_json_limit_is_accepted(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        # /process may take max_upload_bytes even when the JSON limit is far smaller.
        engine = _engine(make_result())
        client = _client(engine, max_upload_bytes=300_000, max_iml_chars=10, max_text_chars=10)

        resp = client.post("/process", files=_upload(wav_bytes + b"\x00" * 250_000))

        assert resp.status_code == 200
        engine.process_voice_input.assert_called_once()

    def test_synthesize_text_too_long_is_422(self) -> None:
        engine = _engine()

        resp = _client(engine, max_text_chars=100).post("/synthesize", json={"text": "a" * 101})

        assert resp.status_code == 422
        engine.synthesize_speech.assert_not_called()

    def test_synthesize_empty_text_is_422(self) -> None:
        engine = _engine()

        resp = _client(engine).post("/synthesize", json={"text": ""})

        assert resp.status_code == 422
        engine.synthesize_speech.assert_not_called()

    def test_generate_iml_too_long_is_422(self) -> None:
        engine = _engine()
        iml = "<utterance>" + "a" * 300 + "</utterance>"

        resp = _client(engine, max_iml_chars=200).post("/generate", json={"iml": iml})

        assert resp.status_code == 422
        engine.generate_response.assert_not_called()


class TestImlValidation:
    @pytest.mark.parametrize("iml", ["", "garbage", "<iml/>", "<utterance emotion='sad'>x"])
    def test_invalid_iml_is_422_without_calling_the_llm(self, iml: str) -> None:
        engine = _engine()

        resp = _client(engine).post("/generate", json={"iml": iml})

        assert resp.status_code == 422
        engine.generate_response.assert_not_called()


# -- Authentication --


class TestAuthentication:
    KEY = "correct horse battery staple"

    def _client(self, engine: MagicMock | None = None, **kwargs: Any) -> Any:
        return _client(engine, api_key=self.KEY, **kwargs)

    def test_open_by_default(self) -> None:
        assert _client().post("/synthesize", json={"text": "hi"}).status_code == 200

    @pytest.mark.parametrize(
        ("method", "path", "kwargs"),
        [
            ("post", "/generate", {"json": {"iml": VALID_IML}}),
            ("post", "/synthesize", {"json": {"text": "hi"}}),
            ("post", "/process", {"files": {"audio": ("a.wav", b"RIFF")}}),
        ],
    )
    def test_endpoints_require_the_key(
        self, method: str, path: str, kwargs: dict[str, Any]
    ) -> None:
        engine = _engine()
        client = self._client(engine)

        assert getattr(client, method)(path, **kwargs).status_code == 401
        wrong = getattr(client, method)(path, headers={"X-API-Key": "nope"}, **kwargs)
        assert wrong.status_code == 401
        engine.process_voice_input.assert_not_called()
        engine.generate_response.assert_not_called()
        engine.synthesize_speech.assert_not_called()

    @pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
    def test_only_health_is_exempt(self, path: str) -> None:
        # The API description is as private as the endpoints.
        client = self._client()

        assert client.get(path).status_code == 401
        assert client.get(path, headers={"X-API-Key": self.KEY}).status_code == 200

    def test_correct_key_is_accepted(self) -> None:
        resp = self._client().post(
            "/synthesize", json={"text": "hi"}, headers={"X-API-Key": self.KEY}
        )

        assert resp.status_code == 200

    def test_non_ascii_key_matches_a_utf8_header(self) -> None:
        key = "cl\u00e9-secr\u00e8te"
        client = _client(api_key=key)

        ok = client.post("/synthesize", json={"text": "hi"}, headers={"X-API-Key": key.encode()})

        assert ok.status_code == 200

    def test_health_stays_open_for_load_balancers(self) -> None:
        assert self._client().get("/health").status_code == 200

    def test_key_is_checked_before_the_size_limit(self) -> None:
        # An unauthenticated client learns nothing about limits and cannot
        # make the server look at its body.  This request is over the size
        # middleware's limit, so a 401 (and not a 413) shows that the key was
        # checked first.
        engine = _engine()
        client = self._client(engine, max_upload_bytes=1000)

        resp = client.post("/process", files=_upload(b"\x00" * 200_000))

        assert resp.status_code == 401
        engine.process_voice_input.assert_not_called()

    def test_health_stays_open_behind_a_path_prefix(self) -> None:
        # Under uvicorn --root-path /api, scope["path"] is "/api/health" while
        # the router sees "/health"; the exemption has to use the latter.
        client = self._prefixed_client()

        assert client.get("/api/health").status_code == 200
        assert client.post("/api/synthesize", json={"text": "hi"}).status_code == 401

    @pytest.mark.parametrize("path", ["/api/api/health", "/apihealth", "/api/health/x"])
    def test_a_prefix_does_not_widen_the_exemption(self, path: str) -> None:
        assert self._prefixed_client().get(path).status_code in (401, 404)

    def test_upload_limit_applies_to_process_behind_a_path_prefix(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        # 3 MB is over the JSON limit (about 2.5 MB) and far under the upload limit.
        from starlette.testclient import TestClient

        client = TestClient(create_app(engine=_engine(make_result())), root_path="/api")

        resp = client.post("/api/process", files=_upload(wav_bytes + b"\x00" * 3_000_000))

        assert resp.status_code == 200

    def _prefixed_client(self) -> Any:
        from starlette.testclient import TestClient

        app = create_app(engine=_engine(), api_key=self.KEY)
        return TestClient(app, root_path="/api", raise_server_exceptions=False)


# -- Error mapping: client mistakes are 4xx, failures upstream 502, nothing leaks --


class _StubSTT(STTProvider):
    """An STT provider that fails, or answers, without a network or a model."""

    def __init__(self, *, error: Exception | None = None, text: str = "hello there") -> None:
        self._error = error
        self._text = text

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        if self._error is not None:
            raise self._error
        return TranscriptionResult(text=self._text, alignments=[], language="en")


def _real_engine(stt: STTProvider) -> IntentEngine:
    """The real ``IntentEngine`` (real orchestration, analysis and IML) over a stub STT."""
    with (
        patch("intent_engine.engine.create_stt_provider", return_value=stt),
        patch("intent_engine.engine.create_llm_provider", return_value=MagicMock()),
        patch("intent_engine.engine.create_tts_provider", return_value=MagicMock()),
    ):
        return IntentEngine()


# Passes the container check (RIFF....WAVE) but is not decodable audio.
CORRUPT_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00garbage"


class TestRealEngineOnUndecodableAudio:
    """What a client sees for an upload the format check passes but no decoder can read.

    ``IntentEngine`` never raises ``AudioProcessingError``: an STT failure is
    an ``STTError`` (a 502, like any provider failure) and a prosody analysis
    failure degrades to text-only IML (a 200).  These tests drive the real
    engine, so they show what the API answers and not what a mock says.
    """

    def test_stt_failing_to_read_the_audio_is_a_502_without_internal_paths(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # What WhisperSTT raises when ffmpeg cannot load the file.
        error = STTError(
            "Whisper transcription failed: RuntimeError: "
            "Failed to load audio: /tmp/tmpfyzx9vwt/audio.wav"
        )
        client = _client(_real_engine(_StubSTT(error=error)))

        with caplog.at_level(logging.WARNING):
            resp = client.post("/process", files=_upload(CORRUPT_WAV))

        assert resp.status_code == 502
        assert "/tmp" not in resp.text
        assert "/tmp/tmpfyzx9vwt/audio.wav" in caplog.text  # the operator still gets the cause

    def test_prosody_that_cannot_read_the_audio_gives_a_text_only_answer(self) -> None:
        pytest.importorskip("numpy")
        pytest.importorskip("parselmouth")
        client = _client(_real_engine(_StubSTT(text="hello there")))

        resp = client.post("/process", files=_upload(CORRUPT_WAV))

        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "hello there"
        assert data["prosody_features"] == []
        assert (data["emotion"], data["confidence"]) == ("neutral", 0.0)
        assert "<prosody" not in data["iml"]


class TestErrorMapping:
    def test_audio_processing_error_from_a_custom_engine_is_422_without_internal_paths(
        self, wav_bytes: bytes
    ) -> None:
        # IntentEngine never raises this (see TestRealEngineOnUndecodableAudio),
        # but the app takes any engine object, and one that does raises a client error.
        engine = _engine()
        engine.process_voice_input = AsyncMock(
            side_effect=AudioProcessingError("Cannot read audio file /tmp/tmpfyzx9vwt/audio.wav")
        )

        resp = _client(engine).post("/process", files=_upload(wav_bytes))

        assert resp.status_code == 422
        assert "/tmp" not in resp.text

    def test_provider_failure_is_502_without_provider_details(
        self, wav_bytes: bytes, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine = _engine()
        engine.process_voice_input = AsyncMock(
            side_effect=STTError("HTTP 401 from https://api.deepgram.com/v1/listen key ...a91f")
        )

        with caplog.at_level(logging.ERROR):
            resp = _client(engine).post("/process", files=_upload(wav_bytes))

        assert resp.status_code == 502
        for leaked in ("deepgram", "401", "a91f"):
            assert leaked not in resp.text
        assert "deepgram" in caplog.text  # the operator still gets the real cause

    def test_error_responses_carry_a_matching_log_reference(
        self, wav_bytes: bytes, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine = _engine()
        engine.process_voice_input = AsyncMock(side_effect=STTError("boom"))

        with caplog.at_level(logging.ERROR):
            resp = _client(engine).post("/process", files=_upload(wav_bytes))

        error_id = resp.json()["detail"].split("error id ")[1].rstrip(").")
        assert error_id and error_id in caplog.text

    def test_llm_failure_is_502(self) -> None:
        engine = _engine()
        engine.generate_response = AsyncMock(
            side_effect=LLMError("LLM interpretation failed: 401 Unauthorized: invalid x-api-key")
        )

        resp = _client(engine).post("/generate", json={"iml": VALID_IML})

        assert resp.status_code == 502
        assert "x-api-key" not in resp.text

    def test_tts_failure_is_502(self) -> None:
        engine = _engine()
        engine.synthesize_speech = AsyncMock(side_effect=TTSError("quota exceeded for key abc"))

        resp = _client(engine).post("/synthesize", json={"text": "Hello"})

        assert resp.status_code == 502
        assert "abc" not in resp.text

    def test_iml_rejected_by_the_engine_is_422(self) -> None:
        engine = _engine()
        engine.generate_response = AsyncMock(side_effect=IMLParseError("bad xml at line 1"))

        resp = _client(engine).post("/generate", json={"iml": VALID_IML})

        assert resp.status_code == 422

    @pytest.mark.parametrize("exc", [RuntimeError("secret detail"), IntentEngineError("secret")])
    def test_unexpected_failures_are_generic_500s(self, exc: Exception) -> None:
        engine = _engine()
        engine.synthesize_speech = AsyncMock(side_effect=exc)

        resp = _client(engine).post("/synthesize", json={"text": "Hello"})

        assert resp.status_code == 500
        assert "secret" not in resp.text


# -- The event loop stays free --


class TestEventLoop:
    async def test_health_answers_while_a_request_is_in_flight(
        self, make_result: MakeResult, wav_bytes: bytes
    ) -> None:
        import httpx

        engine = _engine()

        async def slow(path: str) -> Result:
            await asyncio.sleep(0.6)
            return make_result()

        engine.process_voice_input = slow
        app = create_app(engine=engine)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pending = asyncio.create_task(client.post("/process", files=_upload(wav_bytes)))
            await asyncio.sleep(0.1)
            started = time.monotonic()
            health = await client.get("/health")
            elapsed = time.monotonic() - started
            assert (await pending).status_code == 200

        assert health.status_code == 200
        assert elapsed < 0.3

    def test_upload_is_written_off_the_event_loop_thread(
        self, make_result: MakeResult, wav_bytes: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loop_threads: list[int] = []
        writer_threads: list[int] = []
        real_write = Path.write_bytes

        def spy(self: Path, data: bytes) -> int:
            writer_threads.append(threading.get_ident())
            return real_write(self, data)

        async def process(path: str) -> Result:
            loop_threads.append(threading.get_ident())  # runs on the event loop
            return make_result()

        engine = _engine()
        engine.process_voice_input = process
        monkeypatch.setattr(Path, "write_bytes", spy)

        _client(engine).post("/process", files=_upload(wav_bytes))

        assert writer_threads and loop_threads
        assert loop_threads[0] not in writer_threads

    def test_iml_is_validated_off_the_event_loop_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loop_threads: list[int] = []
        validator_threads: list[int] = []
        real_validate = IMLValidator.validate

        def spy(self: IMLValidator, iml_string: str) -> Any:
            validator_threads.append(threading.get_ident())
            return real_validate(self, iml_string)

        async def generate(iml: str, context: str | None = None, tone: str | None = None) -> Any:
            loop_threads.append(threading.get_ident())  # runs on the event loop
            return Response(text="ok", emotion="neutral")

        engine = _engine()
        engine.generate_response = generate
        monkeypatch.setattr(IMLValidator, "validate", spy)

        _client(engine).post("/generate", json={"iml": VALID_IML})

        assert validator_threads and loop_threads
        assert loop_threads[0] not in validator_threads
