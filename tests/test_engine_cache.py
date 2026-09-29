"""Result cache: keyed by audio content and active profile, bounded, thread-safe.

The pipeline is mocked (fast, deterministic) except for the IML assembler,
which is real so that a prosody profile visibly changes the result.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from prosody_protocol import ProsodyMapping, ProsodyProfile

from intent_engine.engine import IntentEngine
from intent_engine.models.result import Result
from intent_engine.stt.base import TranscriptionResult
from tests.conftest import create_mocked_engine, make_flat_speech

CALM_ON_FLAT = ProsodyProfile(
    profile_version="1.0.0",
    user_id="user-1",
    description=None,
    mappings=(ProsodyMapping({"pitch_contour": "flat"}, "calm", 0.6),),
)
SAD_ON_FLAT = ProsodyProfile(
    profile_version="1.0.0",
    user_id="user-2",
    description=None,
    mappings=(ProsodyMapping({"pitch_contour": "flat"}, "sad", 0.7),),
)


def _engine(**kwargs) -> IntentEngine:
    """Engine with a mocked STT/analyzer of monotone speech and a real assembler."""
    engine = create_mocked_engine(**kwargs)
    alignments, features = make_flat_speech()
    engine._stt.transcribe = AsyncMock(
        return_value=TranscriptionResult(
            text="I am fine thank you today.", alignments=alignments, language="en"
        )
    )
    engine._analyzer.analyze = MagicMock(return_value=features)
    engine._analyzer.detect_pauses = MagicMock(return_value=[])
    return engine


def _audio(tmp_path: Path, name: str = "a.wav", content: bytes = b"RIFF fake audio") -> str:
    path = tmp_path / name
    path.write_bytes(content)
    return str(path)


def _run(engine: IntentEngine, path: str, **kwargs) -> Result:
    return asyncio.run(engine.process_voice_input(path, **kwargs))


class TestProfileAwareKey:
    def test_set_profile_is_not_served_the_result_from_before(self, tmp_path: Path) -> None:
        engine = _engine()
        path = _audio(tmp_path)

        before = _run(engine, path)
        engine.set_profile(CALM_ON_FLAT)
        after = _run(engine, path)

        assert (before.emotion, before.confidence) == ("neutral", 0.0)
        assert (after.emotion, after.confidence) == ("calm", 0.6)
        assert engine._stt.transcribe.call_count == 2

    def test_clear_profile_is_not_served_the_result_from_before(self, tmp_path: Path) -> None:
        engine = _engine()
        path = _audio(tmp_path)
        engine.set_profile(CALM_ON_FLAT)
        with_profile = _run(engine, path)

        engine.clear_profile()
        without = _run(engine, path)

        assert with_profile.emotion == "calm"
        assert without.emotion == "neutral"

    def test_switching_between_profiles_never_crosses_results(self, tmp_path: Path) -> None:
        engine = _engine()
        path = _audio(tmp_path)

        engine.set_profile(CALM_ON_FLAT)
        first = _run(engine, path)
        engine.set_profile(SAD_ON_FLAT)
        second = _run(engine, path)
        engine.set_profile(CALM_ON_FLAT)
        third = _run(engine, path)

        assert [first.emotion, second.emotion, third.emotion] == ["calm", "sad", "calm"]
        # going back to a profile reuses what was computed under it
        assert engine._stt.transcribe.call_count == 2

    def test_same_audio_and_profile_is_a_hit(self, tmp_path: Path) -> None:
        engine = _engine()
        engine.set_profile(CALM_ON_FLAT)
        path = _audio(tmp_path)

        _run(engine, path)
        _run(engine, path)

        assert engine._stt.transcribe.call_count == 1

    def test_profile_from_constructor_path_is_part_of_the_key(self, tmp_path: Path) -> None:
        profile_path = tmp_path / "profile.json"
        profile_path.write_text(
            '{"profile_version": "1.0.0", "user_id": "u", "prosody_mappings": '
            '[{"pattern": {"pitch_contour": "flat"}, '
            '"interpretation": {"emotion": "calm", "confidence_boost": 0.6}}]}',
            encoding="utf-8",
        )
        engine = _engine(prosody_profile=str(profile_path))
        path = _audio(tmp_path)

        with_profile = _run(engine, path)
        engine.clear_profile()
        without = _run(engine, path)

        assert (with_profile.emotion, without.emotion) == ("calm", "neutral")


class TestKeyIsAudioContent:
    def test_same_bytes_at_another_path_hit(self, tmp_path: Path) -> None:
        engine = _engine()

        _run(engine, _audio(tmp_path, "a.wav", b"same bytes"))
        _run(engine, _audio(tmp_path, "b.wav", b"same bytes"))

        assert engine._stt.transcribe.call_count == 1

    def test_changed_bytes_at_the_same_path_miss(self, tmp_path: Path) -> None:
        engine = _engine()
        path = _audio(tmp_path, "a.wav", b"first take")
        _run(engine, path)

        _audio(tmp_path, "a.wav", b"second take")
        _run(engine, path)

        assert engine._stt.transcribe.call_count == 2


class TestDisabledCache:
    @pytest.mark.parametrize("size", [0, -1])
    def test_non_positive_size_disables_caching(self, tmp_path: Path, size: int) -> None:
        engine = _engine(cache_size=size)
        path = _audio(tmp_path)

        first = _run(engine, path)
        second = _run(engine, path)

        assert first.text == second.text == "I am fine thank you today."
        assert len(engine._cache) == 0
        assert engine._stt.transcribe.call_count == 2

    def test_disabled_cache_does_not_read_the_file_for_a_hash(self, tmp_path: Path) -> None:
        engine = _engine(cache_size=0)
        engine._audio_hash = MagicMock(side_effect=AssertionError("hashed"))  # type: ignore[method-assign]

        _run(engine, _audio(tmp_path))


class TestBoundsAndPurging:
    def test_size_is_a_bound(self, tmp_path: Path) -> None:
        engine = _engine(cache_size=2)

        for i in range(5):
            _run(engine, _audio(tmp_path, f"{i}.wav", f"take {i}".encode()))

        assert len(engine._cache) == 2

    def test_clear_cache_forgets_results(self, tmp_path: Path) -> None:
        engine = _engine()
        path = _audio(tmp_path)
        _run(engine, path)

        engine.clear_cache()
        _run(engine, path)

        assert engine._stt.transcribe.call_count == 2

    def test_clear_cache_empties_it(self, tmp_path: Path) -> None:
        engine = _engine()
        _run(engine, _audio(tmp_path))
        assert len(engine._cache) == 1

        engine.clear_cache()

        assert len(engine._cache) == 0


class TestCachedResultsAreIndependent:
    def test_mutating_a_returned_feature_list_does_not_change_later_hits(
        self, tmp_path: Path
    ) -> None:
        engine = _engine()
        path = _audio(tmp_path)

        first = _run(engine, path)
        expected = len(first.prosody_features)
        assert expected > 0
        first.prosody_features.clear()  # a caller tidying up

        second = _run(engine, path)
        second.prosody_features.clear()
        third = _run(engine, path)

        assert len(second.prosody_features) == 0
        assert len(third.prosody_features) == expected
        assert engine._stt.transcribe.call_count == 1

    def test_hits_are_equal_results(self, tmp_path: Path) -> None:
        engine = _engine()
        path = _audio(tmp_path)

        assert _run(engine, path) == _run(engine, path)


@pytest.fixture()
def fast_thread_switching() -> Iterator[None]:
    """Make the interpreter switch threads very often, to expose races."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(old)


class _SlowFirstLookup(OrderedDict):  # type: ignore[type-arg]
    """An LRU dict whose first lookup dawdles.

    It gives another thread a wide window to act between that lookup and
    whatever the caller does with its answer.
    """

    def __init__(self, entered: threading.Event) -> None:
        super().__init__()
        self._entered = entered
        self._slow = True

    def _dawdle(self) -> None:
        if self._slow:
            self._slow = False
            self._entered.set()
            time.sleep(0.05)

    def __contains__(self, key: object) -> bool:
        found = super().__contains__(key)
        self._dawdle()
        return found

    def get(self, key: object, default: object = None) -> object:
        found = super().get(key, default)
        self._dawdle()
        return found

    def __getitem__(self, key: object) -> object:
        found = super().__getitem__(key)
        self._dawdle()
        return found


class TestConcurrency:
    def test_eviction_between_check_and_access_is_not_an_error(self) -> None:
        engine = _engine(cache_size=1)
        entered = threading.Event()
        engine._cache = _SlowFirstLookup(entered)
        engine._cache["a"] = MagicMock(spec=Result)
        errors: list[BaseException] = []

        def reader() -> None:
            try:
                engine._cache_get("a")
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        thread = threading.Thread(target=reader)
        thread.start()
        assert entered.wait(timeout=5)
        engine._cache_put("b", MagicMock(spec=Result))  # evicts "a" mid-lookup
        thread.join(timeout=5)

        assert errors == []

    def test_threads_hammering_the_cache_stay_consistent(
        self, fast_thread_switching: None
    ) -> None:
        engine = _engine(cache_size=2)
        result = MagicMock(spec=Result)
        errors: list[BaseException] = []
        start = threading.Barrier(8)

        def worker(n: int) -> None:
            try:
                start.wait()
                for i in range(1500):
                    key = str((n + i) % 5)
                    engine._cache_get(key)
                    engine._cache_put(key, result)
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in [pool.submit(worker, n) for n in range(8)]:
                future.result()

        assert errors == []
        assert len(engine._cache) <= 2

    def test_concurrent_tasks_on_one_engine_all_get_results(self, tmp_path: Path) -> None:
        engine = _engine(cache_size=2)
        paths = [_audio(tmp_path, f"{i}.wav", f"take {i % 3}".encode()) for i in range(12)]

        async def main() -> list[Result]:
            return list(await asyncio.gather(*(engine.process_voice_input(p) for p in paths)))

        results = asyncio.run(main())

        assert len(results) == 12
        assert {r.text for r in results} == {"I am fine thank you today."}
        assert len(engine._cache) <= 2

    def test_sync_wrapper_threads_share_one_engine(self, tmp_path: Path) -> None:
        engine = _engine(cache_size=2)
        paths = [_audio(tmp_path, f"{i}.wav", f"take {i}".encode()) for i in range(16)]

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(engine.process_voice_input_sync, paths))

        assert len(results) == 16
        assert len(engine._cache) <= 2
