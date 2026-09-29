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
from intent_engine.errors import IntentEngineError, LLMError, STTError, TTSError
from intent_engine.models.audio import Audio
from intent_engine.models.response import Response
from intent_engine.models.result import Result

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
        # make the server look at its body.
        engine = _engine()
        client = self._client(engine, max_upload_bytes=1000)

        resp = client.post("/process", files=_upload(b"\x00" * 200_000))

        assert resp.status_code == 401
        engine.process_voice_input.assert_not_called()


# -- Error mapping: client mistakes are 4xx, failures upstream 502, nothing leaks --


class TestErrorMapping:
    def test_undecodable_audio_is_422_without_internal_paths(self, wav_bytes: bytes) -> None:
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
