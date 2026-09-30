"""LocalEngine -- fully local deployment.

Everything runs on the user's own infrastructure with no network
calls.  Validates that all required models are available on disk
at construction time.  All ``prosody_protocol`` components
(parselmouth, librosa) run locally by default.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from intent_engine._deployment import (
    LOCAL_STT_PROVIDERS,
    LOCAL_TTS_PROVIDERS,
    expand_model_path,
    is_local_llm,
    is_model_file,
    llama_cpp_model_file,
    llm_kwargs_with_model,
    stt_kwargs_with_model,
    tts_kwargs_with_model,
)
from intent_engine.engine import IntentEngine

logger = logging.getLogger(__name__)

# Default provider selections for local mode
_LOCAL_DEFAULTS = {
    "stt_provider": "whisper-prosody",
    "llm_provider": "local",
    "tts_provider": "espeak",
}

# Hardware tier definitions for documentation / validation hints
HARDWARE_TIERS = {
    "minimum": {"ram_gb": 16, "gpu": "CPU-only", "note": "Slow"},
    "recommended": {"ram_gb": 32, "gpu": "NVIDIA RTX 4090", "note": "Good"},
    "optimal": {"ram_gb": 128, "gpu": "2x NVIDIA A100", "note": "Best"},
}


class LocalEngine(IntentEngine):
    """Fully local deployment -- no data leaves the network.

    Inherits the full ``IntentEngine`` pipeline and defaults to local
    providers.  Validates that model files exist on disk before
    proceeding.  :attr:`is_fully_local` reports whether the configured
    providers really keep everything local; a cloud provider is
    accepted but logged as a warning.

    Parameters
    ----------
    stt_provider:
        Local STT provider (``"whisper-prosody"``).
    stt_model:
        Whisper model name (``"tiny"``, ``"base"``, ``"small"``,
        ``"medium"``, ``"large-v3"``, ...) or path to a checkpoint.
        Passed as the adapter's ``model_size``.
    llm_provider:
        Local LLM provider (``"local"``).
    llm_model:
        Path to a local GGUF model (llama.cpp) or, with
        ``llm_kwargs={"base_url": ...}``, a model name on a local
        OpenAI-compatible server such as Ollama or vLLM.
    tts_provider:
        Local TTS provider (``"coqui"`` or ``"espeak"``).
    tts_model:
        Coqui model name or id (e.g. ``"tts_models/en/vctk/vits"``);
        needs ``tts_provider="coqui"``, as eSpeak takes no model.
    prosody_model:
        Informational label for the prosody analyzer variant
        (prosody analysis always uses ``prosody_protocol``).
    constitutional_rules:
        Optional path to a YAML file with constitutional rules.
    prosody_profile:
        Optional path to a prosody profile JSON.
    cache_size:
        Maximum number of audio results to cache.
    validate_models:
        If ``True`` (default), raise ``FileNotFoundError`` when
        a model file (a path, or a name ending in ``.gguf``, ``.pt``, ...)
        does not exist on disk.  That covers ``stt_model``, ``llm_model``,
        ``tts_model`` and a ``model_path`` given in ``llm_kwargs``; a
        leading ``~`` is the home directory.  A Whisper size that Whisper does
        not know is only found out when the model is first loaded.
    stt_kwargs:
        Additional keyword arguments for the STT adapter.
    llm_kwargs:
        Additional keyword arguments for the LLM adapter.
    tts_kwargs:
        Additional keyword arguments for the TTS adapter.
    """

    def __init__(
        self,
        stt_provider: str = _LOCAL_DEFAULTS["stt_provider"],
        stt_model: str | None = None,
        llm_provider: str = _LOCAL_DEFAULTS["llm_provider"],
        llm_model: str | None = None,
        tts_provider: str = _LOCAL_DEFAULTS["tts_provider"],
        tts_model: str | None = None,
        prosody_model: str | None = None,
        constitutional_rules: str | os.PathLike[str] | None = None,
        prosody_profile: str | os.PathLike[str] | None = None,
        cache_size: int = 128,
        validate_models: bool = True,
        stt_kwargs: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        tts_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # Validate model paths exist on disk when they look like file paths
        if validate_models:
            self._validate_model_paths(
                stt_model=stt_model,
                llm_model=llm_model,
                tts_model=tts_model,
            )

        # Hand each model to the parameter its provider really takes
        stt_kw = stt_kwargs_with_model(stt_provider, stt_model, stt_kwargs)
        llm_kw = llm_kwargs_with_model(llm_provider, llm_model, llm_kwargs)
        tts_kw = tts_kwargs_with_model(tts_provider, tts_model, tts_kwargs)

        # A llama.cpp file that came in through llm_kwargs (llm_model, when
        # given, has replaced it above and was checked already)
        model_file = llama_cpp_model_file(llm_provider, llm_kw)
        if validate_models and model_file is not None and not Path(model_file).exists():
            raise FileNotFoundError(f"llm_kwargs model_path does not exist: {model_file}")

        super().__init__(
            stt_provider=stt_provider,
            llm_provider=llm_provider,
            tts_provider=tts_provider,
            constitutional_rules=constitutional_rules,
            prosody_profile=prosody_profile,
            cache_size=cache_size,
            stt_kwargs=stt_kw,
            llm_kwargs=llm_kw,
            tts_kwargs=tts_kw,
        )

        self._stt_model = stt_model
        self._llm_model = llm_model
        self._tts_model = tts_model
        self._prosody_model = prosody_model
        self._is_fully_local = (
            stt_provider in LOCAL_STT_PROVIDERS
            and tts_provider in LOCAL_TTS_PROVIDERS
            and is_local_llm(llm_provider, llm_kw)
        )
        if not self._is_fully_local:
            logger.warning(
                "LocalEngine is not fully local (stt=%s, llm=%s, tts=%s): "
                "audio, transcripts or replies may leave this machine or network",
                stt_provider,
                llm_provider,
                tts_provider,
            )

        logger.info(
            "LocalEngine initialized (stt=%s/%s, llm=%s/%s, tts=%s/%s, prosody=%s)",
            stt_provider,
            stt_model or "default",
            llm_provider,
            llm_model or "default",
            tts_provider,
            tts_model or "default",
            prosody_model or "prosody-protocol",
        )

    @staticmethod
    def _validate_model_paths(
        stt_model: str | None,
        llm_model: str | None,
        tts_model: str | None,
    ) -> None:
        """Validate that model files exist on disk.

        Only checks values that are files: absolute or explicitly relative
        paths (``~`` is the home directory), and names ending in a model
        extension (``.gguf``, ``.pt``, ...).  Model names and ids such as
        ``"large-v3"`` or Coqui's ``"tts_models/en/vctk/vits"`` are not
        validated.
        """
        for label, path in [
            ("stt_model", stt_model),
            ("llm_model", llm_model),
            ("tts_model", tts_model),
        ]:
            if path is None:
                continue
            if is_model_file(path) and not Path(expand_model_path(path)).exists():
                raise FileNotFoundError(
                    f"{label} path does not exist: {path}"
                )

    @property
    def stt_model(self) -> str | None:
        """STT model name or path."""
        return self._stt_model

    @property
    def llm_model(self) -> str | None:
        """LLM model name or path."""
        return self._llm_model

    @property
    def tts_model(self) -> str | None:
        """TTS model name or path."""
        return self._tts_model

    @property
    def prosody_model(self) -> str | None:
        """Prosody analyzer model label."""
        return self._prosody_model

    @property
    def is_fully_local(self) -> bool:
        """Whether the configured providers keep all processing local.

        ``True`` when the STT and TTS providers run on this machine and the
        LLM is a local model or a server on this machine or a private
        network (see ``is_local_url``); ``False`` if any component is a
        cloud service.
        """
        return self._is_fully_local

    @property
    def deployment_mode(self) -> str:
        """Return the deployment mode identifier (``"local"``)."""
        return "local"
