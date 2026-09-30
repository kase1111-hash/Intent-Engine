"""The ``*_sync`` wrappers' background loop: races with close(), escaping
``BaseException``s and reference cycles.

The shared loop thread must never leave a caller blocked forever, must survive
a provider raising ``SystemExit``/``KeyboardInterrupt``, and must not keep the
engine alive after a failed call.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import threading
import time
import weakref
from collections.abc import Callable
from concurrent.futures import CancelledError
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from intent_engine.engine import IntentEngine
from intent_engine.llm.base import InterpretationResult
from intent_engine.stt.base import TranscriptionResult
from tests.conftest import (
    create_mocked_engine,
    make_flat_speech,
    make_interpretation_result,
    make_synthesis_result,
)


def _engine() -> IntentEngine:
    engine = create_mocked_engine()
    alignments, features = make_flat_speech()
    engine._stt.transcribe = AsyncMock(
        return_value=TranscriptionResult(
            text="I am fine thank you today.", alignments=alignments, language="en"
        )
    )
    engine._analyzer.analyze = MagicMock(return_value=features)
    engine._analyzer.detect_pauses = MagicMock(return_value=[])
    engine._llm.interpret = AsyncMock(return_value=make_interpretation_result())
    engine._tts.synthesize = AsyncMock(return_value=make_synthesis_result())
    return engine


def _in_thread(name: str, fn: Callable[[], Any]) -> tuple[threading.Thread, list[Any]]:
    """Run *fn* on a daemon thread; the outcome (value or exception) lands in the list."""
    outcome: list[Any] = []

    def target() -> None:
        try:
            outcome.append(fn())
        except BaseException as exc:  # noqa: BLE001 - inspected by the test
            outcome.append(exc)

    thread = threading.Thread(target=target, name=name, daemon=True)
    thread.start()
    return thread, outcome


def _finish(thread: threading.Thread, outcome: list[Any], timeout: float = 10.0) -> Any:
    thread.join(timeout)
    assert not thread.is_alive(), "the *_sync call never returned"
    assert len(outcome) == 1
    return outcome[0]


class TestCallsRacingClose:
    def test_a_call_that_reaches_the_loop_during_close_is_not_left_hanging(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _engine()
        started = threading.Event()
        in_submit = threading.Event()

        async def never_finishes(iml: str, context: str | None = None) -> InterpretationResult:
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                await asyncio.sleep(0.6)  # slow to cancel: teardown is still running ...
                raise
            return make_interpretation_result()

        engine._llm.interpret = never_finishes
        first, first_outcome = _in_thread("A", lambda: engine.generate_response_sync("<iml/>"))
        assert started.wait(timeout=5)

        real_submit = asyncio.run_coroutine_threadsafe

        def slow_submit(coro: Any, loop: Any) -> Any:
            if threading.current_thread().name == "B":
                in_submit.set()
                time.sleep(0.3)  # ... when this caller, who already has the loop, submits
            return real_submit(coro, loop)

        monkeypatch.setattr("intent_engine.engine.asyncio.run_coroutine_threadsafe", slow_submit)
        second, second_outcome = _in_thread("B", lambda: engine.generate_response_sync("<iml/>"))
        assert in_submit.wait(timeout=5)

        engine.close()

        # B is either served or cancelled together with A, but it does return
        b = _finish(second, second_outcome)
        assert isinstance(b, CancelledError) or getattr(b, "text", None), repr(b)
        a = _finish(first, first_outcome)
        assert isinstance(a, CancelledError), repr(a)
        engine.close()

    def test_a_call_after_close_starts_a_new_loop(self) -> None:
        engine = _engine()
        engine.generate_response_sync("<iml/>")
        engine.close()

        assert engine.generate_response_sync("<iml/>").text
        engine.close()

    def test_hammering_close_never_strands_a_caller(self) -> None:
        engine = _engine()
        stop = threading.Event()

        def caller() -> int:
            done = 0
            while not stop.is_set():
                with contextlib.suppress(CancelledError):  # cancelled by a close() in flight
                    engine.generate_response_sync("<iml/>")
                done += 1
            return done

        workers = [_in_thread(f"w{i}", caller) for i in range(4)]
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            engine.close()
            time.sleep(0.001)
        stop.set()

        for thread, outcome in workers:
            assert isinstance(_finish(thread, outcome), int)
        engine.close()


class TestEscapingBaseExceptions:
    @pytest.mark.parametrize("exc_type", [SystemExit, KeyboardInterrupt])
    def test_the_engine_keeps_working_after_a_provider_raises_it(
        self, exc_type: type[BaseException]
    ) -> None:
        engine = _engine()
        calls = {"n": 0}

        async def interpret(iml: str, context: str | None = None) -> InterpretationResult:
            calls["n"] += 1
            if calls["n"] == 1:
                raise exc_type(3) if exc_type is SystemExit else exc_type()
            return make_interpretation_result()

        engine._llm.interpret = interpret

        with pytest.raises(exc_type):
            engine.generate_response_sync("<iml/>")
        runner = engine._sync_runner
        # straight away, so the teardown race is exercised too
        thread, outcome = _in_thread("next", lambda: engine.generate_response_sync("<iml/>"))
        response = _finish(thread, outcome)

        assert getattr(response, "text", None), repr(response)
        assert runner is not None and runner.thread.is_alive()
        assert engine._sync_runner is runner  # the same loop kept serving
        engine.close()

    def test_repeated_escapes_do_not_accumulate_threads(self) -> None:
        engine = _engine()

        async def interpret(iml: str, context: str | None = None) -> InterpretationResult:
            raise SystemExit(1)

        engine._llm.interpret = interpret
        before = threading.active_count()
        for _ in range(5):
            thread, outcome = _in_thread("next", lambda: engine.generate_response_sync("<iml/>"))
            assert isinstance(_finish(thread, outcome), SystemExit)
        assert threading.active_count() <= before + 1
        engine.close()

    def test_a_runner_whose_thread_died_is_replaced(self) -> None:
        engine = _engine()
        engine.generate_response_sync("<iml/>")
        runner = engine._sync_runner
        assert runner is not None
        runner._request_stop()  # something stopped the loop behind the engine's back
        runner.thread.join(timeout=5)
        assert not runner.thread.is_alive()

        thread, outcome = _in_thread("next", lambda: engine.generate_response_sync("<iml/>"))
        response = _finish(thread, outcome)

        assert getattr(response, "text", None), repr(response)
        assert engine._sync_runner is not runner
        engine.close()

    def test_a_coroutine_that_could_not_be_scheduled_is_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import warnings

        engine = _engine()

        def refuse(coro: Any, loop: Any) -> Any:
            raise RuntimeError("Event loop is closed")

        monkeypatch.setattr("intent_engine.engine.asyncio.run_coroutine_threadsafe", refuse)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError, match="Event loop is closed"):
                engine.generate_response_sync("<iml/>")
            gc.collect()
        monkeypatch.undo()
        engine.close()

        assert not [w for w in caught if "never awaited" in str(w.message)], caught


class TestFailedCallsDoNotKeepTheEngineAlive:
    @pytest.mark.parametrize("failing", ["tts", "llm"])
    def test_engine_is_freed_without_a_garbage_collection(self, failing: str) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("provider down")  # a fresh exception per call

        engine = _engine()
        if failing == "tts":
            engine._tts.synthesize = AsyncMock(side_effect=boom)
        else:
            engine._llm.interpret = AsyncMock(side_effect=boom)

        was_enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        try:
            with pytest.raises(Exception, match="failed"):
                if failing == "tts":
                    engine.synthesize_speech_sync("Hello")
                else:
                    engine.generate_response_sync("<iml/>")
            runner = engine._sync_runner
            assert runner is not None
            ref = weakref.ref(engine)
            del engine
            runner.thread.join(timeout=5)

            assert ref() is None, "the failed call's exception still references the engine"
            assert not runner.thread.is_alive()
        finally:
            if was_enabled:
                gc.enable()
