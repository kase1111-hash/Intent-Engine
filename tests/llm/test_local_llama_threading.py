"""LocalLLM's llama.cpp path must keep the event loop free and serialise the model.

Loading a GGUF file takes tens of seconds and a generation takes seconds, so both
have to run in a worker thread; a ``Llama`` object is not thread-safe, so the
worker threads take turns.  The tests use a fake ``llama_cpp`` module and
threading events instead of timing: a fake that blocks until the *event loop*
releases it can only finish if the loop was free while it waited.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import types
from typing import Any

import pytest

from intent_engine.llm.base import InterpretationResult
from intent_engine.llm.local import LocalLLM

# Only bounds how long a failing test waits; a passing test never reaches it.
_STUCK_S = 2.0


def _reply(intent: str = "greet") -> dict[str, Any]:
    content = json.dumps(
        {"intent": intent, "response_text": "Hello!", "suggested_emotion": "calm"}
    )
    return {"choices": [{"message": {"content": content}}]}


class _Recorder:
    """What the fake model saw: threads, overlap and the number of loads."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.threads: list[tuple[str, int]] = []
        self.loads = 0
        self.active = {"load": 0, "infer": 0}
        self.max_active = {"load": 0, "infer": 0}
        self.overlap = {"load": threading.Event(), "infer": threading.Event()}

    def enter(self, phase: str, overlap_window: float = 0.0) -> None:
        with self.lock:
            self.threads.append((phase, threading.get_ident()))
            self.active[phase] += 1
            self.max_active[phase] = max(self.max_active[phase], self.active[phase])
            if self.active[phase] > 1:
                self.overlap[phase].set()
        if overlap_window:
            # Give a second thread the chance to arrive.  Without the adapter's lock it
            # does, and it is seen; with the lock this only waits out the window.
            self.overlap[phase].wait(overlap_window)

    def leave(self, phase: str) -> None:
        with self.lock:
            self.active[phase] -= 1


def _install_llama_cpp(
    monkeypatch: pytest.MonkeyPatch,
    recorder: _Recorder,
    *,
    overlap_window: float = 0.0,
    infer: Any = None,
) -> None:
    class FakeLlama:
        def __init__(self, **kwargs: object) -> None:
            recorder.enter("load", overlap_window)
            with recorder.lock:
                recorder.loads += 1
            recorder.leave("load")

        def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            recorder.enter("infer", overlap_window)
            try:
                if infer is not None:
                    return infer(kwargs)  # type: ignore[no-any-return]
                return _reply()
            finally:
                recorder.leave("infer")

    module = types.ModuleType("llama_cpp")
    module.Llama = FakeLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", module)


IML = "<utterance>hello</utterance>"


async def test_load_and_inference_run_in_worker_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    _install_llama_cpp(monkeypatch, recorder)

    result = await LocalLLM(model_path="/m.gguf").interpret(IML)

    assert result == InterpretationResult("greet", "Hello!", "calm")
    assert [phase for phase, _ in recorder.threads] == ["load", "infer"]
    loop_thread = threading.get_ident()
    assert all(thread != loop_thread for _, thread in recorder.threads)


async def test_the_loop_stays_free_during_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = threading.Event()
    release = threading.Event()

    def infer(_: dict[str, Any]) -> dict[str, Any]:
        entered.set()
        # Only the event loop (below) can set `release`.  If this ran on the loop,
        # nothing could, so the assertion is what a blocked loop looks like.
        assert release.wait(_STUCK_S), "inference blocked the event loop"
        return _reply()

    _install_llama_cpp(monkeypatch, _Recorder(), infer=infer)
    task = asyncio.create_task(LocalLLM(model_path="/m.gguf").interpret(IML))

    while not entered.is_set():
        await asyncio.sleep(0.001)
    release.set()

    assert (await task).intent == "greet"


async def test_the_loop_stays_free_during_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = threading.Event()
    release = threading.Event()
    recorder = _Recorder()
    _install_llama_cpp(monkeypatch, recorder)
    llama_cpp = sys.modules["llama_cpp"]
    real_init = llama_cpp.Llama.__init__  # type: ignore[attr-defined]

    def slow_init(self: object, **kwargs: object) -> None:
        entered.set()
        assert release.wait(_STUCK_S), "the model load blocked the event loop"
        real_init(self, **kwargs)

    llama_cpp.Llama.__init__ = slow_init  # type: ignore[attr-defined,method-assign]
    task = asyncio.create_task(LocalLLM(model_path="/m.gguf").interpret(IML))

    while not entered.is_set():
        await asyncio.sleep(0.001)
    release.set()

    assert (await task).intent == "greet"


async def test_concurrent_calls_load_once_and_never_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    _install_llama_cpp(monkeypatch, recorder, overlap_window=0.1)
    llm = LocalLLM(model_path="/m.gguf")

    results = await asyncio.gather(*(llm.interpret(IML) for _ in range(3)))

    assert [r.intent for r in results] == ["greet"] * 3
    assert recorder.loads == 1
    assert recorder.max_active == {"load": 1, "infer": 1}


async def test_a_failed_load_does_not_leave_the_lock_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    llm = LocalLLM(model_path="/m.gguf")
    with pytest.raises(ImportError, match="llama-cpp-python is required"):
        await llm.interpret(IML)

    _install_llama_cpp(monkeypatch, _Recorder())
    assert (await asyncio.wait_for(llm.interpret(IML), _STUCK_S)).intent == "greet"


async def test_a_cancelled_call_does_not_disturb_the_next_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling stops the wait, not the generation; the next call still gets its own reply."""
    first_running = threading.Event()
    release_first = threading.Event()

    def infer(kwargs: dict[str, Any]) -> dict[str, Any]:
        prompt = kwargs["messages"][-1]["content"]
        if "first" in prompt:
            first_running.set()
            assert release_first.wait(_STUCK_S), "inference blocked the event loop"
            return _reply("first_intent")
        return _reply("second_intent")

    _install_llama_cpp(monkeypatch, _Recorder(), infer=infer)
    llm = LocalLLM(model_path="/m.gguf")

    first = asyncio.create_task(llm.interpret("<utterance>first</utterance>"))
    while not first_running.is_set():
        await asyncio.sleep(0.001)
    second = asyncio.create_task(llm.interpret("<utterance>second</utterance>"))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release_first.set()

    assert (await asyncio.wait_for(second, _STUCK_S)).intent == "second_intent"


def test_repeated_asyncio_run_on_one_instance_loads_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lock is a threading lock, so it is not tied to the loop that first used it."""
    recorder = _Recorder()
    _install_llama_cpp(monkeypatch, recorder)
    llm = LocalLLM(model_path="/m.gguf")

    results = [asyncio.run(llm.interpret(IML)) for _ in range(3)]

    assert [r.intent for r in results] == ["greet"] * 3
    assert recorder.loads == 1
