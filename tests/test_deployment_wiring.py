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

from intent_engine._deployment import is_local_url, is_model_file, llm_kwargs_with_model
from intent_engine.hybrid_engine import HybridEngine
from intent_engine.local_engine import LocalEngine
from tests.conftest import create_mocked_engine

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


class TestNumericHostsAreReadAsAddresses:
    """The resolver reads a lone number (decimal, hex, octal) as an IPv4 address."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://134744072/v1",  # 8.8.8.8
            "http://0x08080808/v1",  # 8.8.8.8
            "http://010002004010/v1",  # 64.8.8.8
            "http://8.8.8/v1",  # short dotted form: 8.8.0.8
            "http://0x8.0x8.0x8.0x8/v1",
        ],
    )
    def test_public_addresses_in_unusual_notation_are_not_local(self, url: str) -> None:
        assert not is_local_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://2130706433/v1",  # 127.0.0.1
            "http://0x7f000001/v1",  # 127.0.0.1
            "http://3232235521/v1",  # 192.168.0.1
            "http://127.1/v1",
            "http://10.1/v1",
            "http://0177.0.0.1/v1",
        ],
    )
    def test_private_addresses_in_unusual_notation_are_local(self, url: str) -> None:
        assert is_local_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://ollama/v1",
            "http://my-service:8000/v1",
            "http://db2/v1",
            "http://12abc/v1",
            "http://deadbeef/v1",
            "http://0xg/v1",
            "http://a1/v1",
        ],
    )
    def test_service_names_stay_local(self, url: str) -> None:
        assert is_local_url(url)

    def test_the_engines_do_not_call_a_public_number_local(self) -> None:
        llm = {"base_url": "http://134744072/v1"}

        assert LocalEngine(llm_model="llama3", llm_kwargs=llm).is_fully_local is False
        assert HybridEngine(stt_kwargs=FAKE_KEY, llm_kwargs=llm).is_llm_local is False


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home directory holding ``models/m.gguf`` and ``w.pt``."""
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m.gguf").write_bytes(b"GGUF")
    (tmp_path / "w.pt").write_bytes(b"pt")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


class TestHomeDirectoryPaths:
    def test_a_tilde_llm_path_is_expanded_for_the_check_and_for_llama_cpp(
        self, home: Path
    ) -> None:
        engine = LocalEngine(llm_model="~/models/m.gguf")

        assert engine._llm._model_path == str(home / "models" / "m.gguf")
        assert engine.llm_model == "~/models/m.gguf"  # what the caller gave

    def test_a_tilde_stt_path_reaches_whisper_expanded(self, home: Path) -> None:
        engine = LocalEngine(stt_model="~/w.pt", llm_kwargs={"base_url": "http://127.0.0.1:1/v1"})

        assert engine._stt._model_size == str(home / "w.pt")

    def test_a_tilde_model_path_in_llm_kwargs_is_expanded(self, home: Path) -> None:
        engine = LocalEngine(llm_kwargs={"model_path": "~/models/m.gguf"})

        assert engine._llm._model_path == str(home / "models" / "m.gguf")

    def test_hybrid_expands_it_too(self, home: Path) -> None:
        engine = HybridEngine(stt_kwargs=FAKE_KEY, llm_model="~/models/m.gguf")

        assert engine._llm._model_path == str(home / "models" / "m.gguf")

    def test_a_missing_file_under_home_is_still_reported(self, home: Path) -> None:
        with pytest.raises(FileNotFoundError, match="llm_model"):
            LocalEngine(llm_model="~/models/nope.gguf")

    def test_a_server_model_name_is_not_expanded(self) -> None:
        kw = llm_kwargs_with_model("local", "~odd", {"base_url": "http://127.0.0.1:1/v1"})

        assert kw["model"] == "~odd"


class TestUnknownProviderWithAModel:
    def test_local_engine_reports_the_unknown_provider(self) -> None:
        local = {"base_url": "http://127.0.0.1:1/v1"}
        with pytest.raises(ValueError, match="Unknown STT provider"):
            LocalEngine(stt_provider="bogus", stt_model="x", llm_kwargs=local)
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LocalEngine(llm_provider="bogus", llm_model="x")
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            LocalEngine(tts_provider="bogus", tts_model="x", llm_kwargs=local)

    def test_hybrid_engine_reports_the_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            HybridEngine(stt_kwargs=FAKE_KEY, llm_provider="bogus", llm_model="x")

    def test_a_known_provider_without_models_still_says_so(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            LocalEngine(tts_model="coqui-tts-v1", llm_kwargs={"base_url": "http://127.0.0.1:1/v1"})


class TestFailFast:
    def test_a_missing_llama_cpp_file_given_in_llm_kwargs_is_reported(self) -> None:
        with pytest.raises(FileNotFoundError, match="model_path"):
            LocalEngine(llm_kwargs={"model_path": "/nonexistent/x.gguf"})

    def test_the_check_can_be_switched_off(self) -> None:
        engine = LocalEngine(
            llm_kwargs={"model_path": "/nonexistent/x.gguf"}, validate_models=False
        )

        assert engine._llm._model_path == "/nonexistent/x.gguf"

    def test_llm_model_wins_over_a_stale_model_path_in_llm_kwargs(self, gguf: str) -> None:
        engine = LocalEngine(llm_model=gguf, llm_kwargs={"model_path": "/nonexistent/old.gguf"})

        assert engine._llm._model_path == gguf

    def test_a_server_url_makes_model_path_irrelevant(self) -> None:
        LocalEngine(
            llm_kwargs={"base_url": "http://127.0.0.1:1/v1", "model_path": "/nonexistent/x.gguf"}
        )

    def test_hybrid_reports_a_missing_llama_cpp_file(self) -> None:
        with pytest.raises(FileNotFoundError, match="llm_model"):
            HybridEngine(stt_kwargs=FAKE_KEY, llm_model="/nonexistent/x.gguf")
        with pytest.raises(FileNotFoundError, match="model_path"):
            HybridEngine(stt_kwargs=FAKE_KEY, llm_kwargs={"model_path": "/nonexistent/x.gguf"})

    def test_hybrid_check_can_be_switched_off(self) -> None:
        engine = HybridEngine(
            stt_kwargs=FAKE_KEY, llm_model="/nonexistent/x.gguf", validate_models=False
        )

        assert engine.llm_model == "/nonexistent/x.gguf"

    def test_hybrid_does_not_check_a_cloud_model_id_or_a_server_model(self) -> None:
        HybridEngine(
            stt_kwargs=FAKE_KEY,
            llm_provider="claude",
            llm_model="a-cloud-model.bin",
            llm_kwargs=FAKE_KEY,
        )
        HybridEngine(
            stt_kwargs=FAKE_KEY,
            llm_model="llama3",
            llm_kwargs={"base_url": "http://localhost:11434/v1"},
        )

    @pytest.mark.parametrize("size", [None, "5", 2.5, [8]])
    def test_cache_size_must_be_an_integer(self, size: object) -> None:
        with pytest.raises(TypeError, match="cache_size"):
            create_mocked_engine(cache_size=size)

    def test_integer_like_cache_sizes_are_accepted(self) -> None:
        class Size:
            def __index__(self) -> int:
                return 4

        assert create_mocked_engine(cache_size=Size())._cache_size == 4
        assert create_mocked_engine(cache_size=0)._cache_size == 0
        assert create_mocked_engine(cache_size=-1)._cache_size == -1
