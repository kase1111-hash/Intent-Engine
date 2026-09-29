"""Tests for the eSpeak TTS adapter."""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from intent_engine.errors import TTSError
from intent_engine.tts import espeak
from intent_engine.tts.base import EMOTION_VOICE_MAP, SynthesisResult, TTSProvider
from intent_engine.tts.espeak import ESpeakTTS
from tests.tts.helpers import Heartbeat, wav_bytes


class TestESpeakTTSConstruction:
    def test_default_params(self) -> None:
        tts = ESpeakTTS()
        assert tts._voice is None
        assert tts._rate_wpm == 175
        assert tts._volume == 1.0

    def test_custom_params(self) -> None:
        tts = ESpeakTTS(voice="english+f3", rate_wpm=200, volume=0.8)
        assert tts._voice == "english+f3"
        assert tts._rate_wpm == 200
        assert tts._volume == 0.8

    def test_is_tts_provider(self) -> None:
        tts = ESpeakTTS()
        assert isinstance(tts, TTSProvider)

    def test_accepts_kwargs(self) -> None:
        tts = ESpeakTTS(extra_param="ignored")
        assert tts._rate_wpm == 175

    def test_does_not_claim_ssml_support(self) -> None:
        assert ESpeakTTS().supports_ssml is False


class TestESpeakTTSEngine:
    def test_import_error_without_pyttsx3(self) -> None:
        tts = ESpeakTTS()
        with patch.dict(sys.modules, {"pyttsx3": None}), pytest.raises(
            ImportError, match="pyttsx3 is required"
        ):
            tts._create_engine()

    def test_create_engine_sets_voice(self) -> None:
        mock_pyttsx3 = types.ModuleType("pyttsx3")
        mock_engine = MagicMock()
        mock_pyttsx3.init = MagicMock(return_value=mock_engine)  # type: ignore[attr-defined]
        sys.modules["pyttsx3"] = mock_pyttsx3

        try:
            tts = ESpeakTTS(voice="english+f3")
            tts._create_engine()

            mock_engine.setProperty.assert_called_once_with("voice", "english+f3")
        finally:
            sys.modules.pop("pyttsx3", None)

    def test_create_engine_no_voice(self) -> None:
        mock_pyttsx3 = types.ModuleType("pyttsx3")
        mock_engine = MagicMock()
        mock_pyttsx3.init = MagicMock(return_value=mock_engine)  # type: ignore[attr-defined]
        sys.modules["pyttsx3"] = mock_pyttsx3

        try:
            tts = ESpeakTTS()
            tts._create_engine()

            # setProperty should NOT be called for voice when voice is None
            mock_engine.setProperty.assert_not_called()
        finally:
            sys.modules.pop("pyttsx3", None)


class TestESpeakTTSSynthesize:
    def test_synthesize_adjusts_rate_for_emotion(self) -> None:
        mock_pyttsx3 = types.ModuleType("pyttsx3")
        mock_engine = MagicMock()
        mock_pyttsx3.init = MagicMock(return_value=mock_engine)  # type: ignore[attr-defined]
        sys.modules["pyttsx3"] = mock_pyttsx3

        try:
            tts = ESpeakTTS(rate_wpm=175)

            # Mock save_to_file to write a real (if tiny) WAV file
            def fake_save(text: str, path: str) -> None:
                Path(path).write_bytes(wav_bytes())

            mock_engine.save_to_file.side_effect = fake_save

            result = asyncio.run(
                tts.synthesize("I'm angry!", emotion="angry")
            )

            assert isinstance(result, SynthesisResult)
            assert result.format == "wav"

            # Verify rate was adjusted (angry has rate > 1.0)
            rate_calls = [
                call for call in mock_engine.setProperty.call_args_list
                if call.args[0] == "rate"
            ]
            assert len(rate_calls) == 1
            adjusted_rate = rate_calls[0].args[1]
            assert adjusted_rate > 175  # angry = faster
        finally:
            sys.modules.pop("pyttsx3", None)

    def test_synthesize_adjusts_volume_for_emotion(self) -> None:
        mock_pyttsx3 = types.ModuleType("pyttsx3")
        mock_engine = MagicMock()
        mock_pyttsx3.init = MagicMock(return_value=mock_engine)  # type: ignore[attr-defined]
        sys.modules["pyttsx3"] = mock_pyttsx3

        try:
            tts = ESpeakTTS(volume=1.0)

            def fake_save(text: str, path: str) -> None:
                Path(path).write_bytes(wav_bytes())

            mock_engine.save_to_file.side_effect = fake_save

            asyncio.run(
                tts.synthesize("I'm sad", emotion="sad")
            )

            # Verify volume was adjusted (sad has volume_db < 0)
            volume_calls = [
                call for call in mock_engine.setProperty.call_args_list
                if call.args[0] == "volume"
            ]
            assert len(volume_calls) == 1
            adjusted_volume = volume_calls[0].args[1]
            assert adjusted_volume < 1.0  # sad = quieter
        finally:
            sys.modules.pop("pyttsx3", None)

    def test_volume_clamped_to_valid_range(self) -> None:
        mock_pyttsx3 = types.ModuleType("pyttsx3")
        mock_engine = MagicMock()
        mock_pyttsx3.init = MagicMock(return_value=mock_engine)  # type: ignore[attr-defined]
        sys.modules["pyttsx3"] = mock_pyttsx3

        try:
            tts = ESpeakTTS(volume=1.0)

            def fake_save(text: str, path: str) -> None:
                Path(path).write_bytes(wav_bytes())

            mock_engine.save_to_file.side_effect = fake_save

            # angry has +6dB which would push volume above 1.0
            asyncio.run(
                tts.synthesize("Loud", emotion="angry")
            )

            volume_calls = [
                call for call in mock_engine.setProperty.call_args_list
                if call.args[0] == "volume"
            ]
            assert len(volume_calls) == 1
            adjusted_volume = volume_calls[0].args[1]
            assert 0.0 <= adjusted_volume <= 1.0
        finally:
            sys.modules.pop("pyttsx3", None)

    def test_temp_file_cleaned_up(self) -> None:
        mock_pyttsx3 = types.ModuleType("pyttsx3")
        mock_engine = MagicMock()
        mock_pyttsx3.init = MagicMock(return_value=mock_engine)  # type: ignore[attr-defined]
        sys.modules["pyttsx3"] = mock_pyttsx3

        try:
            tts = ESpeakTTS()

            saved_path = None

            def fake_save(text: str, path: str) -> None:
                nonlocal saved_path
                saved_path = path
                Path(path).write_bytes(wav_bytes())

            mock_engine.save_to_file.side_effect = fake_save

            asyncio.run(
                tts.synthesize("Test")
            )

            # Temp file should have been cleaned up
            assert saved_path is not None
            assert not Path(saved_path).exists()
        finally:
            sys.modules.pop("pyttsx3", None)


class FakeEngine:
    """Stand-in for a pyttsx3 engine whose behaviour tests can dial.

    ``save_to_file()`` only records the request, as in pyttsx3; the file is
    written (or not) when ``runAndWait()`` runs.

    Parameters
    ----------
    write:
        Called with the output path to produce the file, or ``None`` to
        write nothing.
    notify:
        Fire ``finished-utterance`` once done, as every pyttsx3 driver does.
    background:
        Return from ``runAndWait()`` immediately and finish on another
        thread after ``delay`` seconds.  pyttsx3 2.99 with espeak-ng
        behaves like this: the file appears once eSpeak has synthesised.
    delay:
        Seconds between ``runAndWait()`` starting and the work finishing.
    block:
        Seconds ``runAndWait()`` blocks the calling thread with
        ``time.sleep``, like a CPU-bound engine.
    """

    active = 0
    max_active = 0
    _guard = threading.Lock()

    def __init__(
        self,
        write: Callable[[str], None] | None = lambda path: Path(path).write_bytes(wav_bytes()),
        notify: bool = True,
        background: bool = False,
        delay: float = 0.0,
        block: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self._write = write
        self._notify = notify
        self._background = background
        self._delay = delay
        self._block = block
        self._error = error
        self.properties: list[tuple[str, Any]] = []
        self.saved: tuple[str, str] | None = None
        self.callbacks: dict[str, list[Callable[..., None]]] = {}

    def setProperty(self, name: str, value: Any) -> None:
        self.properties.append((name, value))

    def property_value(self, name: str) -> Any:
        return [value for key, value in self.properties if key == name][-1]

    def save_to_file(self, text: str, path: str) -> None:
        self.saved = (text, path)

    def connect(self, topic: str, callback: Callable[..., None]) -> dict[str, Any]:
        self.callbacks.setdefault(topic, []).append(callback)
        return {"topic": topic, "cb": callback}

    def disconnect(self, token: dict[str, Any]) -> None:
        self.callbacks[token["topic"]].remove(token["cb"])

    def _fire(self, topic: str, **kwargs: Any) -> None:
        for callback in list(self.callbacks.get(topic, [])):
            callback(**kwargs)

    def _finish(self) -> None:
        assert self.saved is not None
        time.sleep(self._delay)
        if self._error is not None:
            self._fire("error", exception=self._error)
            return
        if self._write is not None:
            self._write(self.saved[1])
        if self._notify:
            self._fire("finished-utterance", name=None, completed=True)

    def runAndWait(self) -> None:
        with FakeEngine._guard:
            FakeEngine.active += 1
            FakeEngine.max_active = max(FakeEngine.max_active, FakeEngine.active)
        try:
            time.sleep(self._block)
            if self._background:
                threading.Thread(target=self._finish, daemon=True).start()
            else:
                self._finish()
        finally:
            with FakeEngine._guard:
                FakeEngine.active -= 1


@pytest.fixture()
def use_engine(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeEngine], FakeEngine]:
    """Return a function that makes ``pyttsx3.init()`` hand out a FakeEngine."""
    FakeEngine.active = 0
    FakeEngine.max_active = 0

    def install(engine: FakeEngine) -> FakeEngine:
        module = types.ModuleType("pyttsx3")
        module.init = lambda *args, **kwargs: engine  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pyttsx3", module)
        return engine

    return install


class TestESpeakOutputValidation:
    """An engine that produced nothing must be an error, not silent audio."""

    async def test_engine_that_writes_nothing_raises(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        use_engine(FakeEngine(write=None))
        with pytest.raises(TTSError, match="no audio"):
            await ESpeakTTS().synthesize("Hello there")

    async def test_missing_output_file_raises(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        use_engine(FakeEngine(write=lambda path: Path(path).unlink()))
        with pytest.raises(TTSError, match="no audio"):
            await ESpeakTTS().synthesize("Hello there")

    async def test_header_only_wav_raises(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        use_engine(FakeEngine(write=lambda path: Path(path).write_bytes(wav_bytes(frames=0))))
        with pytest.raises(TTSError, match="no audio"):
            await ESpeakTTS().synthesize("...")

    async def test_temp_file_removed_after_failure(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        engine = use_engine(FakeEngine(write=None))
        with pytest.raises(TTSError):
            await ESpeakTTS().synthesize("Hello there")
        assert engine.saved is not None
        assert not Path(engine.saved[1]).exists()

    async def test_old_pyttsx3_without_ffmpeg_points_at_ffmpeg(
        self,
        use_engine: Callable[[FakeEngine], FakeEngine],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        use_engine(FakeEngine(write=None))
        monkeypatch.setattr(espeak, "_pyttsx3_version", lambda: (2, 90))
        monkeypatch.setattr(espeak.shutil, "which", lambda name: None)
        with pytest.raises(TTSError, match="ffmpeg") as excinfo:
            await ESpeakTTS().synthesize("Hello there")
        assert "pyttsx3>=2.99" in str(excinfo.value)

    async def test_current_pyttsx3_does_not_blame_ffmpeg(
        self,
        use_engine: Callable[[FakeEngine], FakeEngine],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        use_engine(FakeEngine(write=None))
        monkeypatch.setattr(espeak, "_pyttsx3_version", lambda: (2, 99))
        monkeypatch.setattr(espeak.shutil, "which", lambda name: None)
        with pytest.raises(TTSError) as excinfo:
            await ESpeakTTS().synthesize("Hello there")
        assert "ffmpeg" not in str(excinfo.value)

    async def test_engine_error_notification_is_reported(
        self,
        use_engine: Callable[[FakeEngine], FakeEngine],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        use_engine(FakeEngine(error=RuntimeError("voice not found")))
        monkeypatch.setattr(espeak, "_TIMEOUT_BASE_S", 0.1)
        monkeypatch.setattr(espeak, "_TIMEOUT_PER_CHAR_S", 0.0)
        with pytest.raises(TTSError, match="voice not found"):
            await ESpeakTTS().synthesize("Hello there")

    async def test_engine_error_is_included_when_no_audio_results(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        engine = use_engine(FakeEngine(write=None))
        original_run = engine.runAndWait

        def run_then_report() -> None:
            original_run()
            engine._fire("error", exception=RuntimeError("no such voice"))

        engine.runAndWait = run_then_report  # type: ignore[method-assign]
        with pytest.raises(TTSError, match="no such voice"):
            await ESpeakTTS().synthesize("Hello there")

    async def test_engine_error_with_good_audio_is_only_logged(
        self,
        use_engine: Callable[[FakeEngine], FakeEngine],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        engine = use_engine(FakeEngine())
        original_run = engine.runAndWait

        def run_then_report() -> None:
            engine._fire("error", exception=RuntimeError("no such voice"))
            original_run()

        engine.runAndWait = run_then_report  # type: ignore[method-assign]
        with caplog.at_level(logging.WARNING, logger="intent_engine.tts.espeak"):
            result = await ESpeakTTS().synthesize("Hello there")
        assert result.audio_data == wav_bytes()
        assert "no such voice" in caplog.text

    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    async def test_blank_text_raises_before_the_engine_is_used(
        self, text: str, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        engine = use_engine(FakeEngine())
        with pytest.raises(TTSError, match="no text"):
            await ESpeakTTS().synthesize(text)
        assert engine.saved is None

    async def test_valid_output_is_returned_unchanged(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        use_engine(FakeEngine())
        result = await ESpeakTTS().synthesize("Hello there")
        assert result.audio_data == wav_bytes()
        assert result.format == "wav"
        assert result.sample_rate == 22050


class TestESpeakCompletion:
    """pyttsx3 2.99 returns from runAndWait() before eSpeak has finished."""

    async def test_waits_for_engine_that_finishes_in_the_background(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        use_engine(FakeEngine(background=True, delay=0.15))
        result = await ESpeakTTS().synthesize("A rather long sentence " * 20)
        assert result.audio_data == wav_bytes()

    async def test_completed_file_is_enough_without_a_notification(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        use_engine(FakeEngine(notify=False))
        result = await ESpeakTTS().synthesize("Hello there")
        assert result.audio_data == wav_bytes()

    async def test_partially_written_file_is_not_returned(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        full = wav_bytes(frames=4000)

        def write_in_two_steps(path: str) -> None:
            with open(path, "wb") as handle:
                handle.write(full[:100])
                handle.flush()
                time.sleep(0.15)
                handle.write(full[100:])

        use_engine(FakeEngine(write=write_in_two_steps, notify=False, background=True))
        result = await ESpeakTTS().synthesize("Hello there")
        assert result.audio_data == full

    async def test_gives_up_when_the_engine_never_finishes(
        self,
        use_engine: Callable[[FakeEngine], FakeEngine],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        use_engine(FakeEngine(write=None, notify=False))
        monkeypatch.setattr(espeak, "_TIMEOUT_BASE_S", 0.1)
        monkeypatch.setattr(espeak, "_TIMEOUT_PER_CHAR_S", 0.0)
        with pytest.raises(TTSError, match="timed out"):
            await ESpeakTTS().synthesize("Hello there")

    async def test_callbacks_are_disconnected_afterwards(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        engine = use_engine(FakeEngine())
        await ESpeakTTS().synthesize("Hello there")
        assert all(not callbacks for callbacks in engine.callbacks.values())


class TestESpeakEmotionMapping:
    """Emotion -> rate/volume must keep the ordering of EMOTION_VOICE_MAP."""

    async def _properties(
        self,
        use_engine: Callable[[FakeEngine], FakeEngine],
        emotion: str,
        **kwargs: Any,
    ) -> tuple[int, float]:
        engine = use_engine(FakeEngine())
        await ESpeakTTS(**kwargs).synthesize("Hello there", emotion=emotion)
        return engine.property_value("rate"), engine.property_value("volume")

    async def _all(
        self, use_engine: Callable[[FakeEngine], FakeEngine], **kwargs: Any
    ) -> dict[str, tuple[int, float]]:
        return {
            emotion: await self._properties(use_engine, emotion, **kwargs)
            for emotion in EMOTION_VOICE_MAP
        }

    async def test_louder_emotions_are_louder_than_neutral(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        volumes = {e: v for e, (_, v) in (await self._all(use_engine)).items()}
        louder = [e for e, p in EMOTION_VOICE_MAP.items() if p.volume_db > 0]
        assert louder  # the table has some (angry, frustrated, joyful, ...)
        for emotion in louder:
            assert volumes[emotion] > volumes["neutral"], emotion

    async def test_quieter_emotions_are_quieter_than_neutral(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        volumes = {e: v for e, (_, v) in (await self._all(use_engine)).items()}
        for emotion, params in EMOTION_VOICE_MAP.items():
            if params.volume_db < 0:
                assert volumes[emotion] < volumes["neutral"], emotion

    async def test_volume_order_follows_the_emotion_table(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        volumes = {e: v for e, (_, v) in (await self._all(use_engine)).items()}
        for a, pa in EMOTION_VOICE_MAP.items():
            for b, pb in EMOTION_VOICE_MAP.items():
                if pa.volume_db > pb.volume_db:
                    assert volumes[a] > volumes[b], (a, b)
                elif pa.volume_db == pb.volume_db:
                    assert volumes[a] == volumes[b], (a, b)

    async def test_volume_differences_match_the_decibel_offsets(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        volumes = {e: v for e, (_, v) in (await self._all(use_engine)).items()}
        # +6 dB is a factor of about two in amplitude
        ratio = volumes["angry"] / volumes["neutral"]
        assert ratio == pytest.approx(10 ** (6 / 20), rel=1e-6)

    async def test_rate_order_follows_the_emotion_table(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        rates = {e: r for e, (r, _) in (await self._all(use_engine)).items()}
        for a, pa in EMOTION_VOICE_MAP.items():
            for b, pb in EMOTION_VOICE_MAP.items():
                if pa.rate > pb.rate:
                    assert rates[a] > rates[b], (a, b)
                elif pa.rate == pb.rate:
                    assert rates[a] == rates[b], (a, b)
        assert rates["neutral"] == 175

    async def test_volumes_stay_within_the_engine_range(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        for volume in (0.0, 0.3, 1.0, 1.7):
            for _, v in (await self._all(use_engine, volume=volume)).values():
                assert 0.0 <= v <= 1.0

    async def test_configured_volume_scales_every_emotion(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        full = await self._all(use_engine, volume=1.0)
        half = await self._all(use_engine, volume=0.5)
        for emotion in EMOTION_VOICE_MAP:
            assert half[emotion][1] == pytest.approx(full[emotion][1] / 2)

    async def test_unusable_emotions_use_the_neutral_settings(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        neutral = await self._properties(use_engine, "neutral")
        for emotion in ("Neutral", "excited", None, ["calm"], 7):
            assert await self._properties(use_engine, emotion) == neutral  # type: ignore[arg-type]

    async def test_emotion_case_is_ignored(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        angry = await self._properties(use_engine, "angry")
        assert await self._properties(use_engine, " ANGRY ") == angry


class TestESpeakEventLoop:
    async def test_event_loop_stays_responsive_while_synthesising(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        use_engine(FakeEngine(block=0.3))
        async with Heartbeat() as heartbeat:
            result = await ESpeakTTS().synthesize("Hello there")
        assert result.audio_data == wav_bytes()
        assert heartbeat.ticks >= 5

    async def test_concurrent_calls_take_turns_on_the_engine(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        # eSpeak keeps one process-wide callback, so two syntheses must not overlap.
        use_engine(FakeEngine(block=0.05))
        results = await asyncio.gather(
            *(ESpeakTTS().synthesize(f"Sentence {i}") for i in range(4))
        )
        assert [r.audio_data for r in results] == [wav_bytes()] * 4
        assert FakeEngine.max_active == 1


class TestESpeakMarkup:
    SSML = (
        '<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xml:lang="en-US"><s><prosody pitch="+5%" volume="+3dB">I am so happy!'
        "</prosody></s></speak>"
    )

    async def test_ssml_document_is_spoken_as_plain_text(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        engine = use_engine(FakeEngine())
        await ESpeakTTS().synthesize(self.SSML, emotion="joyful")
        assert engine.saved is not None
        assert engine.saved[0] == "I am so happy!"

    async def test_plain_text_is_passed_through_untouched(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        engine = use_engine(FakeEngine())
        await ESpeakTTS().synthesize("if a < b then <b>bold</b>")
        assert engine.saved is not None
        assert engine.saved[0] == "if a < b then <b>bold</b>"

    async def test_empty_ssml_document_is_reported_as_no_text(
        self, use_engine: Callable[[FakeEngine], FakeEngine]
    ) -> None:
        use_engine(FakeEngine())
        with pytest.raises(TTSError, match="no text"):
            await ESpeakTTS().synthesize('<speak xml:lang="en-US"><s></s></speak>')
