"""LocalEngine / HybridEngine wire their model options to the real adapters.

The adapters accept (and ignore) unknown keyword arguments, so a misnamed
option silently does nothing.  These tests build the real adapters -- they
load their models lazily, so nothing is downloaded and no network is used --
and check what they were configured with.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from intent_engine._deployment import is_local_url, is_model_file
from intent_engine.hybrid_engine import HybridEngine
from intent_engine.local_engine import LocalEngine

FAKE_KEY = {"api_key": "not-a-real-key"}


@pytest.fixture()
def gguf(tmp_path: Path) -> str:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF")
    return str(path)


class TestLocalEngineModelWiring:
    def test_stt_model_selects_the_whisper_model(self) -> None:
        engine = LocalEngine(
            stt_model="large-v3", llm_kwargs={"base_url": "http://127.0.0.1:1/v1"}
        )

        assert engine._stt._model_size == "large-v3"
        assert engine.stt_model == "large-v3"

    def test_tts_model_selects_the_coqui_model(self) -> None:
        engine = LocalEngine(
            tts_provider="coqui",
            tts_model="tts_models/en/vctk/vits",
            llm_kwargs={"base_url": "http://127.0.0.1:1/v1"},
        )

        assert engine._tts._model_name == "tts_models/en/vctk/vits"

    def test_gguf_llm_model_is_the_llama_cpp_model_path(self, gguf: str) -> None:
        engine = LocalEngine(llm_model=gguf)

        assert engine._llm._model_path == gguf
        assert engine._llm._base_url is None

    def test_llm_model_name_with_a_server_is_the_server_model(self) -> None:
        engine = LocalEngine(
            llm_model="llama3:8b", llm_kwargs={"base_url": "http://localhost:11434/v1"}
        )

        assert engine._llm._model == "llama3:8b"
        assert engine._llm._base_url == "http://localhost:11434/v1"
        assert engine._llm._model_path is None

    def test_llm_model_name_without_a_server_is_an_error_that_says_what_to_do(self) -> None:
        with pytest.raises(ValueError, match="base_url"):
            LocalEngine(llm_model="llama3")

    def test_model_ids_with_slashes_are_not_taken_for_files(self) -> None:
        # Coqui's own default id and Hugging Face style ids contain slashes
        engine = LocalEngine(
            tts_provider="coqui",
            tts_model="tts_models/en/ljspeech/tacotron2-DDC",
            stt_model="openai/whisper-large-v3",
            llm_model="org/model",
            llm_kwargs={"base_url": "http://127.0.0.1:1/v1"},
        )

        assert engine._tts._model_name == "tts_models/en/ljspeech/tacotron2-DDC"
        assert engine._stt._model_size == "openai/whisper-large-v3"

    def test_missing_model_files_are_still_reported(self) -> None:
        with pytest.raises(FileNotFoundError, match="llm_model"):
            LocalEngine(llm_model="/nonexistent/model.gguf")
        with pytest.raises(FileNotFoundError, match="stt_model"):
            LocalEngine(stt_model="/nonexistent/whisper.pt")

    def test_tts_model_is_rejected_by_a_provider_without_models(self) -> None:
        with pytest.raises(ValueError, match="tts_model.*espeak"):
            LocalEngine(tts_model="coqui-tts-v1", llm_kwargs={"base_url": "http://127.0.0.1:1/v1"})

    def test_stt_model_is_rejected_by_a_provider_without_models(self) -> None:
        with pytest.raises(ValueError, match="stt_model.*assemblyai"):
            LocalEngine(
                stt_provider="assemblyai",
                stt_model="x",
                stt_kwargs=FAKE_KEY,
                llm_kwargs={"base_url": "http://127.0.0.1:1/v1"},
            )

    def test_model_option_overrides_the_same_key_in_provider_kwargs(self) -> None:
        engine = LocalEngine(
            stt_model="small",
            stt_kwargs={"model_size": "tiny", "device": "cuda"},
            llm_kwargs={"base_url": "http://127.0.0.1:1/v1"},
        )

        assert engine._stt._model_size == "small"
        assert engine._stt._device == "cuda"


class TestLocalEngineIsFullyLocal:
    LOCAL_LLM = {"base_url": "http://127.0.0.1:1/v1"}

    def test_true_for_local_providers(self, gguf: str) -> None:
        assert LocalEngine(llm_model=gguf).is_fully_local is True
        assert LocalEngine(llm_kwargs=self.LOCAL_LLM).is_fully_local is True

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:11434/v1",
            "http://127.0.0.1:8000/v1",
            "http://[::1]:8000/v1",
            "http://192.168.1.20:8000/v1",
            "http://10.0.0.5/v1",
            "http://ollama:11434/v1",
            "http://gpu-box.internal:8000/v1",
        ],
    )
    def test_true_for_a_server_on_this_machine_or_a_private_network(self, url: str) -> None:
        assert LocalEngine(llm_kwargs={"base_url": url}).is_fully_local is True

    @pytest.mark.parametrize(
        "url",
        ["https://api.example.com/v1", "http://8.8.8.8/v1", "https://llm.example.org:8443/v1"],
    )
    def test_false_for_a_remote_server(self, url: str) -> None:
        assert LocalEngine(llm_kwargs={"base_url": url}).is_fully_local is False

    def test_false_with_a_cloud_stt_provider(self) -> None:
        engine = LocalEngine(
            stt_provider="deepgram", stt_kwargs=FAKE_KEY, llm_kwargs=self.LOCAL_LLM
        )

        assert engine.is_fully_local is False

    def test_false_with_a_cloud_tts_provider(self) -> None:
        engine = LocalEngine(
            tts_provider="elevenlabs", tts_kwargs=FAKE_KEY, llm_kwargs=self.LOCAL_LLM
        )

        assert engine.is_fully_local is False

    def test_false_with_a_cloud_llm_provider(self) -> None:
        engine = LocalEngine(llm_provider="claude", llm_kwargs=FAKE_KEY)

        assert engine.is_fully_local is False

    def test_cloud_components_are_still_wired(self) -> None:
        engine = LocalEngine(
            stt_provider="deepgram",
            stt_model="nova-3",
            stt_kwargs=FAKE_KEY,
            llm_provider="claude",
            llm_model="claude-x",
            llm_kwargs=FAKE_KEY,
        )

        assert engine._stt._model == "nova-3"
        assert engine._llm._model == "claude-x"

    def test_a_non_local_configuration_is_reported_in_the_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="intent_engine"):
            LocalEngine(
                stt_provider="deepgram", stt_kwargs=FAKE_KEY, llm_kwargs=self.LOCAL_LLM
            )

        assert any("not fully local" in r.getMessage() for r in caplog.records)

    def test_a_local_configuration_is_not(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="intent_engine"):
            LocalEngine(llm_kwargs=self.LOCAL_LLM)

        assert not caplog.records


class TestHybridEngineWiring:
    def test_gguf_llm_model_is_the_llama_cpp_model_path(self, gguf: str) -> None:
        engine = HybridEngine(stt_kwargs=FAKE_KEY, llm_model=gguf)

        assert engine._llm._model_path == gguf
        assert engine.llm_model == gguf

    def test_llm_model_name_with_a_server_is_the_server_model(self) -> None:
        engine = HybridEngine(
            stt_kwargs=FAKE_KEY,
            llm_model="llama3",
            llm_kwargs={"base_url": "http://localhost:11434/v1"},
        )

        assert engine._llm._model == "llama3"
        assert engine._llm._model_path is None

    def test_llm_model_name_without_a_server_is_an_error_that_says_what_to_do(self) -> None:
        with pytest.raises(ValueError, match="base_url"):
            HybridEngine(stt_kwargs=FAKE_KEY, llm_model="llama3")

    def test_llm_model_reaches_a_cloud_llm_provider(self) -> None:
        engine = HybridEngine(
            stt_kwargs=FAKE_KEY,
            llm_provider="claude",
            llm_model="a-cloud-model-id",
            llm_kwargs=FAKE_KEY,
        )

        assert engine._llm._model == "a-cloud-model-id"

    def test_is_llm_local_for_a_local_llm(self, gguf: str) -> None:
        assert HybridEngine(stt_kwargs=FAKE_KEY, llm_model=gguf).is_llm_local is True

    @pytest.mark.parametrize(
        ("url", "local"),
        [
            ("http://127.0.0.1:8000/v1", True),
            ("http://192.168.0.7/v1", True),
            ("https://api.example.com/v1", False),
        ],
    )
    def test_is_llm_local_follows_the_server_address(self, url: str, local: bool) -> None:
        engine = HybridEngine(stt_kwargs=FAKE_KEY, llm_kwargs={"base_url": url})

        assert engine.is_llm_local is local

    def test_is_llm_local_is_false_for_a_cloud_llm(self) -> None:
        engine = HybridEngine(
            stt_kwargs=FAKE_KEY, llm_provider="claude", llm_kwargs=FAKE_KEY
        )

        assert engine.is_llm_local is False

    def test_a_non_local_llm_is_reported_in_the_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="intent_engine"):
            HybridEngine(stt_kwargs=FAKE_KEY, llm_provider="claude", llm_kwargs=FAKE_KEY)

        assert any("LLM is not local" in r.getMessage() for r in caplog.records)


class TestPathAndUrlClassification:
    @pytest.mark.parametrize(
        "value",
        [
            "model.gguf",
            "weights/large.pt",
            "/models/large-v3",
            "./models/large-v3",
            "../models/x",
            "~/models/x",
            "C:\\models\\x",
        ],
    )
    def test_file_values(self, value: str) -> None:
        assert is_model_file(value)

    @pytest.mark.parametrize(
        "value",
        [
            "large-v3",
            "llama3:8b",
            "tts_models/en/vctk/vits",
            "openai/whisper-large-v3",
            "org/model",
        ],
    )
    def test_name_values(self, value: str) -> None:
        assert not is_model_file(value)

    @pytest.mark.parametrize("url", ["", "not a url", "http://[::1", "http:///v1", "/v1"])
    def test_unusable_urls_are_not_local(self, url: str) -> None:
        assert not is_local_url(url)

    @pytest.mark.parametrize("url", ["http://169.254.1.1/v1", "http://[fd00::1]:80/v1"])
    def test_link_local_and_unique_local_addresses_are_local(self, url: str) -> None:
        assert is_local_url(url)
