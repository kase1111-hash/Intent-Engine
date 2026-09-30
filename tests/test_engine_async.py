"""Async hygiene: the *_sync wrappers and blocking work off the event loop."""

from __future__ import annotations

import asyncio
import gc
import json
import threading
import time
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from intent_engine.engine import IntentEngine
from intent_engine.llm.base import InterpretationResult, LLMProvider
from intent_engine.stt.base import TranscriptionResult
from tests.conftest import (
    create_mocked_engine,
    make_flat_speech,
    make_interpretation_result,
    make_synthesis_result,
)


def _engine(**kwargs: Any) -> IntentEngine:
    engine = create_mocked_engine(**kwargs)
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


def _audio(tmp_path: Path) -> str:
    path = tmp_path / "a.wav"
    path.write_bytes(b"RIFF fake audio")
    return str(path)


class TestSyncWrappersInsideARunningLoop:
    """Blocking a running loop (Jupyter, FastAPI, bots) is refused, clearly."""

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("process_voice_input_sync", ("AUDIO",)),
            ("generate_response_sync", ("<iml/>",)),
            ("synthesize_speech_sync", ("Hello",)),
            ("type_to_speech_sync", ("Hello",)),
        ],
    )
    def test_raises_runtime_error_pointing_to_the_async_api(
        self, tmp_path: Path, method: str, args: tuple[str, ...]
    ) -> None:
        engine = _engine()
        args = tuple(_audio(tmp_path) if a == "AUDIO" else a for a in args)
        async_name = method.removesuffix("_sync")

        async def call_from_a_running_loop() -> None:
            getattr(engine, method)(*args)

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # e.g. "coroutine ... was never awaited"
            with pytest.raises(RuntimeError, match=rf"await engine\.{async_name}\(") as info:
                asyncio.run(call_from_a_running_loop())
            gc.collect()

        assert method in str(info.value)
        assert "running event loop" in str(info.value)

    def test_the_async_api_still_works_there(self, tmp_path: Path) -> None:
        engine = _engine()

        async def main() -> str:
            return (await engine.process_voice_input(_audio(tmp_path))).text

        assert asyncio.run(main()) == "I am fine thank you today."


class _LoopBoundLLM(LLMProvider):
    """Like the SDK-backed adapters, holds something bound to the first loop it runs on."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None

    async def interpret(self, iml_input: str, context: str | None = None) -> InterpretationResult:
        running = asyncio.get_running_loop()
        if self.loop is None:
            self.loop = running
        elif self.loop is not running:
            raise RuntimeError("Event loop is closed")
        return make_interpretation_result()


class TestSyncWrappersShareOneLoop:
    def test_repeated_calls_run_on_the_same_loop(self) -> None:
        engine = _engine()
        engine._llm = _LoopBoundLLM()

        for _ in range(4):
            assert engine.generate_response_sync("<iml/>").text == "Hello! How can I help you?"

        engine.close()

    def test_calls_from_many_threads_are_all_served(self, tmp_path: Path) -> None:
        engine = _engine()
        engine._llm = _LoopBoundLLM()
        path = _audio(tmp_path)

        with ThreadPoolExecutor(max_workers=6) as pool:
            responses = list(pool.map(lambda _: engine.generate_response_sync("<iml/>"), range(12)))
            results = list(
                pool.map(lambda _: engine.process_voice_input_sync(path, use_cache=False), range(6))
            )

        assert len(responses) == 12 and len(results) == 6
        engine.close()

    def test_errors_propagate_unchanged(self) -> None:
        from intent_engine.errors import TTSError

        engine = _engine()
        engine._tts.synthesize = AsyncMock(side_effect=RuntimeError("TTS down"))

        with pytest.raises(TTSError, match="TTS synthesis failed"):
            engine.synthesize_speech_sync("Hello")
        # and the loop is still usable afterwards
        engine._tts.synthesize = AsyncMock(return_value=make_synthesis_result())
        assert engine.synthesize_speech_sync("Hello").data

        engine.close()

    def test_an_interrupted_wait_cancels_the_running_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cancelled: list[bool] = []

        class InterruptedFuture:
            def result(self) -> None:
                raise KeyboardInterrupt

            def cancel(self) -> None:
                cancelled.append(True)

        def fake_submit(coro: Any, loop: Any) -> InterruptedFuture:
            coro.close()
            return InterruptedFuture()

        engine = _engine()
        monkeypatch.setattr("intent_engine.engine.asyncio.run_coroutine_threadsafe", fake_submit)

        with pytest.raises(KeyboardInterrupt):
            engine.generate_response_sync("<iml/>")

        assert cancelled == [True]
        monkeypatch.undo()
        engine.close()

    def test_close_cancels_calls_still_running(self) -> None:
        from concurrent.futures import CancelledError

        engine = _engine()
        started = threading.Event()

        async def never_finishes(iml: str, context: str | None = None) -> InterpretationResult:
            started.set()
            await asyncio.sleep(60)
            return make_interpretation_result()

        engine._llm.interpret = never_finishes
        outcome: list[BaseException] = []

        def caller() -> None:
            try:
                engine.generate_response_sync("<iml/>")
            except BaseException as exc:  # noqa: BLE001 - inspected below
                outcome.append(exc)

        thread = threading.Thread(target=caller)
        thread.start()
        assert started.wait(timeout=5)

        engine.close()
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert len(outcome) == 1 and isinstance(outcome[0], CancelledError)

    def test_sync_wrappers_work_after_the_caller_used_asyncio_run(self) -> None:
        engine = _engine()
        engine._llm = _LoopBoundLLM()

        asyncio.run(engine.generate_response("<iml/>"))  # binds to that (now closed) loop
        # the wrappers use the engine's own loop, unaffected by the caller's
        engine._llm = _LoopBoundLLM()
        assert engine.generate_response_sync("<iml/>").text
        assert engine.generate_response_sync("<iml/>").text

        engine.close()


class _FakeOpenAIServer:
    """A minimal OpenAI-compatible chat endpoint on 127.0.0.1."""

    def __init__(self) -> None:
        content = json.dumps(
            {"intent": "greet", "response_text": "hello", "suggested_emotion": "calm"}
        )
        body = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                self.rfile.read(int(self.headers.get("content-length", 0)))
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class TestSyncWrappersWithARealAsyncClient:
    def test_consecutive_calls_all_succeed(self) -> None:
        pytest.importorskip("openai")
        from intent_engine.llm.local import LocalLLM

        server = _FakeOpenAIServer()
        engine = _engine()
        try:
            # Each call runs on the engine's loop with a real AsyncOpenAI client
            # (keep-alive connections): every one must succeed, not alternate
            # with "Event loop is closed".
            engine._llm = LocalLLM(base_url=server.url)
            for i in range(4):
                assert engine.generate_response_sync("<utterance>hi</utterance>").text == "hello", i
        finally:
            engine.close()
            server.close()


class TestLifecycle:
    def test_no_thread_until_a_sync_wrapper_is_used(self) -> None:
        engine = _engine()
        asyncio.run(engine.generate_response("<iml/>"))

        assert engine._sync_runner is None

    def test_close_stops_the_loop_thread_and_the_engine_keeps_working(self) -> None:
        engine = _engine()
        engine.generate_response_sync("<iml/>")
        runner = engine._sync_runner
        assert runner is not None and runner.thread.is_alive()

        engine.close()

        assert not runner.thread.is_alive()
        assert engine._sync_runner is None
        engine.close()  # idempotent
        assert engine.generate_response_sync("<iml/>").text  # starts a new loop lazily
        assert engine._sync_runner is not runner
        engine.close()

    def test_dropping_the_engine_stops_the_loop_thread(self) -> None:
        engine = _engine()
        engine.generate_response_sync("<iml/>")
        runner = engine._sync_runner
        assert runner is not None and runner.thread.is_alive()

        ref = weakref.ref(engine)
        del engine
        gc.collect()
        runner.thread.join(timeout=5)

        assert ref() is None
        assert not runner.thread.is_alive()

    def test_a_forked_child_gets_its_own_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import os

        engine = _engine()
        engine.generate_response_sync("<iml/>")
        parents = engine._sync_runner

        real_pid = os.getpid()
        monkeypatch.setattr("intent_engine.engine.os.getpid", lambda: real_pid + 1)
        engine.generate_response_sync("<iml/>")  # would hang on the parent's loop

        assert engine._sync_runner is not parents
        monkeypatch.undo()
        engine.close()
        parents.close()  # the "parent" thread is really ours; stop it too


class TestBlockingWorkLeavesTheLoop:
    def test_hashing_and_analysis_run_in_worker_threads(self, tmp_path: Path) -> None:
        engine = _engine()
        seen: dict[str, int] = {}
        features = engine._analyzer.analyze.return_value
        real_hash = engine._audio_hash

        def analyze(path: str, alignments: Any) -> Any:
            seen["analyze"] = threading.get_ident()
            return features

        def detect_pauses(path: str) -> list[Any]:
            seen["pauses"] = threading.get_ident()
            return []

        def audio_hash(path: str) -> str:
            seen["hash"] = threading.get_ident()
            return real_hash(path)

        engine._analyzer.analyze = analyze
        engine._analyzer.detect_pauses = detect_pauses
        engine._audio_hash = audio_hash  # type: ignore[method-assign]

        async def main() -> int:
            await engine.process_voice_input(_audio(tmp_path))
            return threading.get_ident()

        loop_thread = asyncio.run(main())

        assert set(seen) == {"analyze", "pauses", "hash"}
        assert all(ident != loop_thread for ident in seen.values())

    def test_the_loop_keeps_running_during_slow_gil_releasing_analysis(
        self, tmp_path: Path
    ) -> None:
        # Only work that releases the GIL (reading and decoding audio, numpy)
        # leaves the loop free; Praat holds the GIL, so real analysis still
        # stalls the loop for part of its duration (see process_voice_input).
        engine = _engine()
        features = engine._analyzer.analyze.return_value

        def slow_analyze(path: str, alignments: Any) -> Any:
            time.sleep(0.6)  # releases the GIL
            return features

        engine._analyzer.analyze = slow_analyze

        async def main() -> float:
            ticks: list[float] = []

            async def heartbeat() -> None:
                while True:
                    ticks.append(time.perf_counter())
                    await asyncio.sleep(0.01)

            beat = asyncio.create_task(heartbeat())
            await asyncio.sleep(0.05)
            await engine.process_voice_input(_audio(tmp_path))
            await asyncio.sleep(0.05)  # let the heartbeat tick once more
            beat.cancel()
            return max(b - a for a, b in zip(ticks, ticks[1:], strict=False))

        longest_gap = asyncio.run(main())

        assert longest_gap < 0.3, f"the event loop stalled for {longest_gap:.2f}s"
