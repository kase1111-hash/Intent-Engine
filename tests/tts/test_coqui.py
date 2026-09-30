"""Tests for the Coqui TTS adapter."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
import wave
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from intent_engine.tts.base import SynthesisResult, TTSProvider
from intent_engine.tts.coqui import CoquiTTS, _float_samples_to_wav
from tests.tts.helpers import Heartbeat


class TestCoquiTTSConstruction:
    def test_default_params(self) -> None:
        tts = CoquiTTS()
        assert tts._model_name == "tts_models/en/ljspeech/tacotron2-DDC"
        assert tts._device == "cpu"
        assert tts._speaker is None
        assert tts._language is None
        assert tts._tts is None

    def test_custom_params(self) -> None:
        tts = CoquiTTS(
            model_name="tts_models/en/vctk/vits",
            device="cuda",
            speaker="p225",
            language="en",
        )
        assert tts._model_name == "tts_models/en/vctk/vits"
        assert tts._device == "cuda"
        assert tts._speaker == "p225"
        assert tts._language == "en"

    def test_is_tts_provider(self) -> None:
        tts = CoquiTTS()
        assert isinstance(tts, TTSProvider)

    def test_accepts_kwargs(self) -> None:
        tts = CoquiTTS(extra_param="ignored")
        assert tts._model_name == "tts_models/en/ljspeech/tacotron2-DDC"

    def test_lazy_load_not_triggered_on_init(self) -> None:
        tts = CoquiTTS()
        assert tts._tts is None


class TestCoquiTTSLazyLoad:
    def test_import_error_without_tts(self) -> None:
        tts = CoquiTTS()
        with patch.dict(sys.modules, {"TTS": None, "TTS.api": None}), pytest.raises(
            ImportError, match="TTS .Coqui. is required"
        ):
            tts._load_model()


class TestCoquiTTSSynthesize:
    def test_synthesize_returns_synthesis_result(self) -> None:
        mock_tts_instance = MagicMock()
        # Return a list of float samples
        mock_tts_instance.tts.return_value = [0.0, 0.5, -0.5, 0.3, -0.1]

        tts = CoquiTTS()
        tts._tts = mock_tts_instance

        result = asyncio.run(
            tts.synthesize("Hello world", emotion="calm")
        )

        assert isinstance(result, SynthesisResult)
        assert result.format == "wav"
        assert result.sample_rate == 22050
        assert result.duration is not None
        assert result.duration > 0
        assert len(result.audio_data) > 0

    def test_synthesize_passes_speed_from_emotion(self) -> None:
        mock_tts_instance = MagicMock()
        mock_tts_instance.tts.return_value = [0.0, 0.1]

        tts = CoquiTTS()
        tts._tts = mock_tts_instance

        asyncio.run(
            tts.synthesize("Fast speech", emotion="angry")
        )

        call_kwargs = mock_tts_instance.tts.call_args.kwargs
        # angry has rate > 1.0
        assert call_kwargs["speed"] > 1.0

    def test_synthesize_with_speaker_and_language(self) -> None:
        mock_tts_instance = MagicMock()
        mock_tts_instance.tts.return_value = [0.0]

        tts = CoquiTTS(speaker="p225", language="en")
        tts._tts = mock_tts_instance

        asyncio.run(
            tts.synthesize("Test")
        )

        call_kwargs = mock_tts_instance.tts.call_args.kwargs
        assert call_kwargs["speaker"] == "p225"
        assert call_kwargs["language"] == "en"

    def test_output_is_valid_wav(self) -> None:
        mock_tts_instance = MagicMock()
        mock_tts_instance.tts.return_value = [0.0, 0.5, -0.5, 0.3]

        tts = CoquiTTS()
        tts._tts = mock_tts_instance

        result = asyncio.run(
            tts.synthesize("Hello")
        )

        # Verify it's a valid WAV file
        buf = BytesIO(result.audio_data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 22050
            assert wf.getnframes() == 4


class StubModel:
    """Stand-in for ``TTS.api.TTS`` with the attributes the adapter reads."""

    instances: list[StubModel] = []
    load_delay = 0.0  # seconds the constructor takes, like loading weights

    def __init__(
        self,
        model_name: str = "stub",
        sample_rate: object = 22050,
        samples: int = 100,
        delay: float = 0.0,
    ) -> None:
        time.sleep(StubModel.load_delay)
        self.model_name = model_name
        self.synthesizer = types.SimpleNamespace(output_sample_rate=sample_rate)
        self.samples = samples
        self.delay = delay
        self.device: str | None = None
        self.calls: list[dict[str, Any]] = []
        self.threads: list[threading.Thread] = []
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()
        StubModel.instances.append(self)

    def to(self, device: str) -> StubModel:
        self.device = device
        return self

    def tts(self, **kwargs: Any) -> list[float]:
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.calls.append(kwargs)
            self.threads.append(threading.current_thread())
            time.sleep(self.delay)
            return [0.1] * self.samples
        finally:
            with self._guard:
                self.active -= 1


@pytest.fixture()
def coqui_package(monkeypatch: pytest.MonkeyPatch) -> type[StubModel]:
    """Install a fake ``TTS.api`` module whose ``TTS`` class is StubModel."""
    StubModel.instances = []
    StubModel.load_delay = 0.0
    package = types.ModuleType("TTS")
    api = types.ModuleType("TTS.api")
    api.TTS = StubModel  # type: ignore[attr-defined]
    package.api = api  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "TTS", package)
    monkeypatch.setitem(sys.modules, "TTS.api", api)
    return StubModel


class TestCoquiSampleRate:
    """The WAV must be labelled with the rate the model really produced."""

    async def test_rate_comes_from_the_loaded_model(self) -> None:
        model = StubModel(sample_rate=24000, samples=48000)
        tts = CoquiTTS()
        tts._tts = model

        result = await tts.synthesize("Hello")

        assert result.sample_rate == 24000
        assert result.duration == pytest.approx(2.0)
        with wave.open(BytesIO(result.audio_data), "rb") as wf:
            assert wf.getframerate() == 24000
            assert wf.getnframes() == 48000

    @pytest.mark.parametrize(
        "reported",
        [None, 0, -22050, "24000", True, float("nan"), MagicMock()],
    )
    async def test_unusable_rate_falls_back_to_22050(self, reported: object) -> None:
        tts = CoquiTTS()
        tts._tts = StubModel(sample_rate=reported)

        result = await tts.synthesize("Hello")

        assert result.sample_rate == 22050
        with wave.open(BytesIO(result.audio_data), "rb") as wf:
            assert wf.getframerate() == 22050

    async def test_model_without_a_synthesizer_uses_22050(self) -> None:
        model = StubModel()
        model.synthesizer = None  # type: ignore[assignment]
        tts = CoquiTTS()
        tts._tts = model

        assert (await tts.synthesize("Hello")).sample_rate == 22050

    async def test_float_rate_is_rounded_to_an_int(self) -> None:
        tts = CoquiTTS()
        tts._tts = StubModel(sample_rate=16000.0)

        result = await tts.synthesize("Hello")

        assert result.sample_rate == 16000
        assert isinstance(result.sample_rate, int)


class TestCoquiEventLoop:
    async def test_synthesis_runs_off_the_event_loop_thread(self) -> None:
        model = StubModel()
        tts = CoquiTTS()
        tts._tts = model

        await tts.synthesize("Hello")

        assert model.threads
        assert all(thread is not threading.main_thread() for thread in model.threads)

    async def test_event_loop_stays_responsive_during_synthesis(self) -> None:
        tts = CoquiTTS()
        tts._tts = StubModel(delay=0.3)

        async with Heartbeat() as heartbeat:
            result = await tts.synthesize("Hello")

        assert len(result.audio_data) > 44
        assert heartbeat.ticks >= 5

    async def test_event_loop_stays_responsive_while_the_model_loads(
        self, coqui_package: type[StubModel], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(coqui_package, "load_delay", 0.3)  # restored after the test
        tts = CoquiTTS(model_name="some/model", device="cpu")

        async with Heartbeat() as heartbeat:
            result = await tts.synthesize("Hello")

        assert len(result.audio_data) > 44
        assert heartbeat.ticks >= 5
        assert len(coqui_package.instances) == 1

    async def test_concurrent_first_calls_load_the_model_once_and_take_turns(
        self, coqui_package: type[StubModel]
    ) -> None:
        tts = CoquiTTS(device="cuda")

        results = await asyncio.gather(*(tts.synthesize(f"Sentence {i}") for i in range(4)))

        assert len(results) == 4
        (model,) = coqui_package.instances
        assert model.device == "cuda"
        assert len(model.calls) == 4
        assert model.max_active == 1


class TestCoquiInputs:
    SSML = (
        '<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xml:lang="en-US"><s>I am so happy!</s></speak>'
    )

    def test_does_not_claim_ssml_support(self) -> None:
        assert CoquiTTS().supports_ssml is False

    async def test_ssml_document_is_synthesized_as_plain_text(self) -> None:
        model = StubModel()
        tts = CoquiTTS()
        tts._tts = model

        await tts.synthesize(self.SSML, emotion="joyful")

        assert model.calls[0]["text"] == "I am so happy!"

    async def test_plain_text_is_passed_through_untouched(self) -> None:
        model = StubModel()
        tts = CoquiTTS()
        tts._tts = model

        await tts.synthesize("if a < b then <b>bold</b>")

        assert model.calls[0]["text"] == "if a < b then <b>bold</b>"

    @pytest.mark.parametrize("emotion", [None, ["calm"], "excited", 5])
    async def test_unusable_emotion_uses_neutral_speed(self, emotion: Any) -> None:
        model = StubModel()
        tts = CoquiTTS()
        tts._tts = model

        await tts.synthesize("Hello", emotion=emotion)

        assert model.calls[0]["speed"] == 1.0

    async def test_emotion_case_is_ignored(self) -> None:
        model = StubModel()
        tts = CoquiTTS()
        tts._tts = model

        await tts.synthesize("Hello", emotion="ANGRY")

        assert model.calls[0]["speed"] > 1.0


class TestFloatSamplesToWav:
    def test_empty_samples(self) -> None:
        wav_bytes = _float_samples_to_wav([], 22050)
        buf = BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() == 0

    def test_sample_conversion(self) -> None:
        wav_bytes = _float_samples_to_wav([0.0, 1.0, -1.0], 22050)
        buf = BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() == 3
            assert wf.getsampwidth() == 2

    def test_clamping(self) -> None:
        # Values beyond [-1.0, 1.0] should be clamped
        wav_bytes = _float_samples_to_wav([2.0, -2.0], 22050)
        buf = BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() == 2


class TestCoquiCancellation:
    async def test_a_cancelled_call_does_not_disturb_the_next_one(self) -> None:
        """Cancelling stops the wait, not the synthesis; the next call still gets its own audio."""
        started = threading.Event()
        release = threading.Event()

        class Blocking(StubModel):
            def tts(self, **kwargs: Any) -> list[float]:
                if kwargs["text"] == "first":
                    started.set()
                    assert release.wait(5), "the synthesis was never released"
                    return [0.1] * 10
                return [0.2] * 30

        tts = CoquiTTS()
        tts._tts = Blocking()

        first = asyncio.create_task(tts.synthesize("first"))
        while not started.is_set():
            await asyncio.sleep(0.001)
        second = asyncio.create_task(tts.synthesize("second"))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()

        result = await asyncio.wait_for(second, 5)

        assert result.duration == pytest.approx(30 / 22050)
