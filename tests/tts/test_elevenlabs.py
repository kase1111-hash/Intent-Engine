"""Tests for the ElevenLabs TTS adapter."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
import types
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from intent_engine.tts.base import EMOTION_VOICE_MAP, SynthesisResult, TTSProvider
from intent_engine.tts.elevenlabs import ELEVENLABS_EMOTION_SETTINGS, ElevenLabsTTS
from tests.tts.helpers import SDK_OUTPUT_FORMATS, Heartbeat


class StubClient:
    """Minimal ``ElevenLabs`` client whose ``convert`` behaves like the SDK's.

    The SDK's ``text_to_speech.convert`` is a generator function: calling it
    does nothing, and the HTTP request happens while the result is iterated.
    """

    def __init__(self, chunks: tuple[bytes, ...] = (b"audio",), delay: float = 0.0) -> None:
        self.chunks = chunks
        self.delay = delay
        self.calls: list[dict[str, object]] = []
        self.iterated_on: list[threading.Thread] = []
        self.text_to_speech = types.SimpleNamespace(convert=self._convert)

    def _convert(self, **kwargs: object) -> Iterator[bytes]:
        self.calls.append(kwargs)
        self.iterated_on.append(threading.current_thread())
        time.sleep(self.delay)  # the network round trip
        yield from self.chunks


@pytest.fixture()
def stub_sdk(monkeypatch: pytest.MonkeyPatch) -> StubClient:
    """Install a fake ``elevenlabs`` package handing out one StubClient."""
    client = StubClient()
    module = types.ModuleType("elevenlabs")
    module.ElevenLabs = lambda **kwargs: client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "elevenlabs", module)
    return client


class TestElevenLabsTTSConstruction:
    def test_requires_api_key(self) -> None:
        old_val = os.environ.pop("ELEVENLABS_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="ElevenLabs API key is required"):
                ElevenLabsTTS()
        finally:
            if old_val is not None:
                os.environ["ELEVENLABS_API_KEY"] = old_val

    def test_accepts_explicit_api_key(self) -> None:
        tts = ElevenLabsTTS(api_key="test-key-123")
        assert tts._api_key == "test-key-123"

    def test_reads_env_var(self) -> None:
        os.environ["ELEVENLABS_API_KEY"] = "env-key-456"
        try:
            tts = ElevenLabsTTS()
            assert tts._api_key == "env-key-456"
        finally:
            del os.environ["ELEVENLABS_API_KEY"]

    def test_default_params(self) -> None:
        tts = ElevenLabsTTS(api_key="key")
        assert tts._voice_id == "21m00Tcm4TlvDq8ikWAM"
        assert tts._model_id == "eleven_monolingual_v1"
        assert tts._output_format == "mp3_44100_128"

    def test_custom_params(self) -> None:
        tts = ElevenLabsTTS(
            api_key="key",
            voice_id="custom-voice-id",
            model_id="eleven_multilingual_v2",
            output_format="pcm_22050",
        )
        assert tts._voice_id == "custom-voice-id"
        assert tts._model_id == "eleven_multilingual_v2"
        assert tts._output_format == "pcm_22050"

    def test_is_tts_provider(self) -> None:
        tts = ElevenLabsTTS(api_key="key")
        assert isinstance(tts, TTSProvider)

    def test_accepts_kwargs(self) -> None:
        tts = ElevenLabsTTS(api_key="key", extra_param="ignored")
        assert tts._api_key == "key"


class TestElevenLabsEmotionSettings:
    CORE_EMOTIONS = [
        "neutral", "sincere", "sarcastic", "frustrated", "joyful",
        "uncertain", "angry", "sad", "fearful", "surprised",
        "disgusted", "calm", "empathetic",
    ]

    def test_all_core_emotions_mapped(self) -> None:
        for emotion in self.CORE_EMOTIONS:
            assert emotion in ELEVENLABS_EMOTION_SETTINGS, (
                f"Missing ElevenLabs mapping for: {emotion}"
            )

    def test_settings_have_required_keys(self) -> None:
        for emotion, settings in ELEVENLABS_EMOTION_SETTINGS.items():
            assert "stability" in settings, f"{emotion} missing stability"
            assert "similarity_boost" in settings, f"{emotion} missing similarity_boost"
            assert "style" in settings, f"{emotion} missing style"

    def test_values_in_range(self) -> None:
        for emotion, settings in ELEVENLABS_EMOTION_SETTINGS.items():
            assert 0.0 <= settings["stability"] <= 1.0, (
                f"{emotion} stability out of range"
            )
            assert 0.0 <= settings["similarity_boost"] <= 1.0, (
                f"{emotion} similarity_boost out of range"
            )
            assert 0.0 <= settings["style"] <= 1.0, (
                f"{emotion} style out of range"
            )

    def test_angry_has_high_style(self) -> None:
        assert ELEVENLABS_EMOTION_SETTINGS["angry"]["style"] > 0.7

    def test_calm_has_high_stability(self) -> None:
        assert ELEVENLABS_EMOTION_SETTINGS["calm"]["stability"] > 0.6


class TestElevenLabsTTSSynthesize:
    def test_import_error_without_sdk(self) -> None:
        tts = ElevenLabsTTS(api_key="key")
        with patch.dict(sys.modules, {"elevenlabs": None}), pytest.raises(
            ImportError, match="elevenlabs is required"
        ):
            asyncio.run(
                tts.synthesize("Hello")
            )

    def test_synthesize_returns_synthesis_result(self) -> None:
        mock_elevenlabs = types.ModuleType("elevenlabs")

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"audio", b"bytes"])

        mock_elevenlabs.ElevenLabs = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
        sys.modules["elevenlabs"] = mock_elevenlabs

        try:
            tts = ElevenLabsTTS(api_key="test-key")
            result = asyncio.run(
                tts.synthesize("Hello world", emotion="joyful")
            )

            assert isinstance(result, SynthesisResult)
            assert result.audio_data == b"audiobytes"
            assert result.format == "mp3"
            assert result.sample_rate == 44100
        finally:
            sys.modules.pop("elevenlabs", None)

    def test_synthesize_passes_emotion_settings(self) -> None:
        mock_elevenlabs = types.ModuleType("elevenlabs")

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"data"])

        mock_elevenlabs.ElevenLabs = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
        sys.modules["elevenlabs"] = mock_elevenlabs

        try:
            tts = ElevenLabsTTS(api_key="test-key")
            asyncio.run(
                tts.synthesize("I'm frustrated", emotion="frustrated")
            )

            call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
            voice_settings = call_kwargs["voice_settings"]
            expected = ELEVENLABS_EMOTION_SETTINGS["frustrated"]
            assert voice_settings["stability"] == expected["stability"]
            assert voice_settings["similarity_boost"] == expected["similarity_boost"]
            assert voice_settings["style"] == expected["style"]
        finally:
            sys.modules.pop("elevenlabs", None)

    def test_unknown_emotion_uses_neutral_settings(self) -> None:
        mock_elevenlabs = types.ModuleType("elevenlabs")

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"data"])

        mock_elevenlabs.ElevenLabs = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
        sys.modules["elevenlabs"] = mock_elevenlabs

        try:
            tts = ElevenLabsTTS(api_key="test-key")
            asyncio.run(
                tts.synthesize("Test", emotion="unknown_emotion")
            )

            call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
            voice_settings = call_kwargs["voice_settings"]
            expected = ELEVENLABS_EMOTION_SETTINGS["neutral"]
            assert voice_settings["stability"] == expected["stability"]
        finally:
            sys.modules.pop("elevenlabs", None)


class TestElevenLabsEmotionLookup:
    def test_settings_cover_exactly_the_voice_map_vocabulary(self) -> None:
        assert set(ELEVENLABS_EMOTION_SETTINGS) == set(EMOTION_VOICE_MAP)

    @pytest.mark.parametrize("raw", ["Angry", " ANGRY ", "angry\n"])
    async def test_case_and_whitespace_are_ignored(
        self, raw: str, stub_sdk: StubClient
    ) -> None:
        await ElevenLabsTTS(api_key="k").synthesize("Hi", emotion=raw)
        settings = stub_sdk.calls[0]["voice_settings"]
        assert settings["style"] == ELEVENLABS_EMOTION_SETTINGS["angry"]["style"]  # type: ignore[index]

    @pytest.mark.parametrize("raw", [None, ["calm"], {"a": 1}, 3, b"sad", "excited"])
    async def test_unusable_values_use_neutral_settings(
        self, raw: object, stub_sdk: StubClient
    ) -> None:
        await ElevenLabsTTS(api_key="k").synthesize("Hi", emotion=raw)  # type: ignore[arg-type]
        settings = stub_sdk.calls[0]["voice_settings"]
        neutral = ELEVENLABS_EMOTION_SETTINGS["neutral"]
        assert settings["stability"] == neutral["stability"]  # type: ignore[index]
        assert settings["style"] == neutral["style"]  # type: ignore[index]


class TestElevenLabsOutputFormat:
    """``SynthesisResult`` must describe the bytes that were requested."""

    @pytest.mark.parametrize(("output_format", "expected"), sorted(SDK_OUTPUT_FORMATS.items()))
    async def test_result_matches_requested_format(
        self, output_format: str, expected: tuple[str, int], stub_sdk: StubClient
    ) -> None:
        tts = ElevenLabsTTS(api_key="k", output_format=output_format)
        result = await tts.synthesize("Hello")
        assert (result.format, result.sample_rate) == expected
        assert stub_sdk.calls[0]["output_format"] == output_format

    async def test_default_format_is_unchanged(self, stub_sdk: StubClient) -> None:
        result = await ElevenLabsTTS(api_key="k").synthesize("Hello")
        assert (result.format, result.sample_rate) == ("mp3", 44100)

    @pytest.mark.parametrize("output_format", ["", "mystery", "pcm", "pcm_fast", "MP3_44100_128"])
    async def test_unrecognised_format_warns_and_still_synthesizes(
        self,
        output_format: str,
        stub_sdk: StubClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="intent_engine.tts.elevenlabs"):
            tts = ElevenLabsTTS(api_key="k", output_format=output_format)
        assert "output_format" in caplog.text
        result = await tts.synthesize("Hello")
        assert result.audio_data == b"audio"

    def test_recognised_format_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="intent_engine.tts.elevenlabs"):
            ElevenLabsTTS(api_key="k", output_format="pcm_16000")
        assert caplog.text == ""


class TestElevenLabsEventLoop:
    async def test_request_runs_off_the_event_loop_thread(self, stub_sdk: StubClient) -> None:
        await ElevenLabsTTS(api_key="k").synthesize("Hello")
        # convert() is a generator: the request happens when it is iterated
        assert len(stub_sdk.iterated_on) == 1
        assert stub_sdk.iterated_on[0] is not threading.main_thread()

    async def test_event_loop_stays_responsive_during_the_request(
        self, stub_sdk: StubClient
    ) -> None:
        stub_sdk.delay = 0.3
        async with Heartbeat() as heartbeat:
            result = await ElevenLabsTTS(api_key="k").synthesize("Hello")
        assert result.audio_data == b"audio"
        assert heartbeat.ticks >= 5

    async def test_chunks_are_joined_in_order(self, stub_sdk: StubClient) -> None:
        stub_sdk.chunks = (b"one", b"", b"two", b"three")
        result = await ElevenLabsTTS(api_key="k").synthesize("Hello")
        assert result.audio_data == b"onetwothree"

    async def test_sdk_errors_propagate(self, stub_sdk: StubClient) -> None:
        def failing(**kwargs: object) -> Iterator[bytes]:
            raise RuntimeError("401 unauthorized")
            yield b""  # pragma: no cover - makes this a generator like the SDK's

        stub_sdk.text_to_speech.convert = failing  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="401 unauthorized"):
            await ElevenLabsTTS(api_key="k").synthesize("Hello")


class TestElevenLabsMarkup:
    def test_does_not_claim_ssml_support(self) -> None:
        assert ElevenLabsTTS(api_key="k").supports_ssml is False

    async def test_text_is_sent_verbatim(self, stub_sdk: StubClient) -> None:
        # ElevenLabs takes its own inline tags (<break/>), so nothing is rewritten.
        text = 'Wait <break time="1.0s" /> for it'
        await ElevenLabsTTS(api_key="k").synthesize(text)
        assert stub_sdk.calls[0]["text"] == text
