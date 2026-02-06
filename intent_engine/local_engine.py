"""LocalEngine -- fully local deployment.

Everything runs on the user's own infrastructure with no network
calls.  Validates that all required models are available on disk
at construction time.  All ``prosody_protocol`` components
(parselmouth, librosa) run locally by default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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

    Inherits the full ``IntentEngine`` pipeline but uses only
    local providers.  Validates that model paths exist on disk
    before proceeding.

    Parameters
    ----------
    stt_provider:
        Local STT provider (``"whisper-prosody"``).
    stt_model:
        Whisper model name or path (e.g., ``"whisper-large-v3"``).
    llm_provider:
        Local LLM provider (``"local"``).
    llm_model:
        Path to a local GGUF model or Ollama model name.
    tts_provider:
        Local TTS provider (``"coqui"`` or ``"espeak"``).
    tts_model:
        Optional TTS model name or path.
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
        a model path does not exist on disk.
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
        constitutional_rules: str | None = None,
        prosody_profile: str | None = None,
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

        # Merge model paths into provider kwargs
        stt_kw = dict(stt_kwargs or {})
        if stt_model:
            stt_kw["model"] = stt_model

        llm_kw = dict(llm_kwargs or {})
        if llm_model:
            llm_kw["model_path"] = llm_model

        tts_kw = dict(tts_kwargs or {})
        if tts_model:
            tts_kw["model"] = tts_model

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
        """Validate that model paths exist on disk.

        Only checks paths that look like file paths (contain a slash
        or end with common model extensions).  Named models like
        ``"whisper-large-v3"`` are not validated.
        """
        model_extensions = (".gguf", ".bin", ".pt", ".pth", ".onnx", ".safetensors")

        for label, path in [
            ("stt_model", stt_model),
            ("llm_model", llm_model),
            ("tts_model", tts_model),
        ]:
            if path is None:
                continue
            is_filepath = "/" in path or "\\" in path or path.endswith(model_extensions)
            if is_filepath and not Path(path).exists():
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
        """Whether all processing is local (always ``True``)."""
        return True

    @property
    def deployment_mode(self) -> str:
        """Return the deployment mode identifier."""
        return "local"
