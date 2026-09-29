"""Shared wiring for the deployment engines (``LocalEngine``, ``HybridEngine``).

The provider adapters ignore keyword arguments they do not know, so an
option passed under the wrong name would silently do nothing.  These helpers
map the engines' ``*_model`` options onto the parameter each adapter really
takes (or refuse when it takes none), and work out whether a configuration
keeps data on the user's own machine or network.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlparse

#: File extensions that mark a model value as a file on disk.
MODEL_EXTENSIONS = (".gguf", ".bin", ".pt", ".pth", ".onnx", ".safetensors")

#: Providers that run on the user's own machine.
LOCAL_STT_PROVIDERS = frozenset({"whisper-prosody"})
LOCAL_LLM_PROVIDERS = frozenset({"local"})
LOCAL_TTS_PROVIDERS = frozenset({"coqui", "espeak"})

# Constructor parameter of each provider that takes a model.
_STT_MODEL_PARAM = {"whisper-prosody": "model_size", "deepgram": "model"}
_LLM_MODEL_PARAM = {"claude": "model", "openai": "model"}
_TTS_MODEL_PARAM = {"coqui": "model_name", "elevenlabs": "model_id"}


def is_model_file(value: str) -> bool:
    """Whether a model option names a file, as opposed to a model name or id.

    Model ids such as Coqui's ``tts_models/en/vctk/vits`` or Hugging Face's
    ``org/model`` contain slashes but are not files; a value is a file when it
    ends in a model file extension or is an absolute or explicitly relative
    (``./``, ``../``, ``~``) path.
    """
    if value.endswith(MODEL_EXTENSIONS):
        return True
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return True
    return value.startswith(("./", "../", "~", ".\\", "..\\"))


def _with_model(
    kind: str,
    provider: str,
    model: str | None,
    kwargs: dict[str, Any] | None,
    params: dict[str, str],
) -> dict[str, Any]:
    kw = dict(kwargs or {})
    if not model:
        return kw
    param = params.get(provider)
    if param is None:
        raise ValueError(
            f"{kind}_model is not supported by the {provider!r} {kind.upper()} provider "
            f"(providers that take one: {', '.join(sorted(params))}); "
            f"leave it out or choose another provider."
        )
    kw[param] = model
    return kw


def stt_kwargs_with_model(
    provider: str, model: str | None, kwargs: dict[str, Any] | None
) -> dict[str, Any]:
    """STT adapter kwargs with *model* under the parameter that provider takes."""
    return _with_model("stt", provider, model, kwargs, _STT_MODEL_PARAM)


def tts_kwargs_with_model(
    provider: str, model: str | None, kwargs: dict[str, Any] | None
) -> dict[str, Any]:
    """TTS adapter kwargs with *model* under the parameter that provider takes."""
    return _with_model("tts", provider, model, kwargs, _TTS_MODEL_PARAM)


def llm_kwargs_with_model(
    provider: str, model: str | None, kwargs: dict[str, Any] | None
) -> dict[str, Any]:
    """LLM adapter kwargs with *model* in the form that provider takes.

    The ``local`` provider runs a llama.cpp model file (``model_path``) or
    talks to an OpenAI-compatible server such as Ollama or vLLM (``base_url``
    in *kwargs*, and the server's model name as ``model``).  Cloud providers
    take a model name as ``model``.
    """
    kw = dict(kwargs or {})
    if not model:
        return kw
    if provider != "local":
        return _with_model("llm", provider, model, kw, _LLM_MODEL_PARAM)
    if kw.get("base_url"):
        kw["model"] = model
    elif is_model_file(model):
        kw["model_path"] = model
    else:
        raise ValueError(
            f"llm_model {model!r} is not a model file (.gguf), so it is taken as a model "
            "name on an OpenAI-compatible server (Ollama, vLLM); give that server's URL "
            "with llm_kwargs={'base_url': 'http://localhost:11434/v1'}."
        )
    return kw


def is_local_url(url: str) -> bool:
    """Whether *url* points at this machine or a private network.

    Loopback, private and link-local addresses, ``localhost``, and host names
    that cannot be public (a single label such as a Docker service name, or
    ``.local``, ``.internal``, ``.lan``, ``.home.arpa``) count as local.  A
    best-effort reading of the address only: no name is resolved.
    """
    try:
        host = urlparse(url).hostname
    except ValueError:
        return False
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return "." not in host or host.endswith((".local", ".internal", ".lan", ".home.arpa"))
    return address.is_loopback or address.is_private or address.is_link_local


def is_local_llm(provider: str, kwargs: dict[str, Any]) -> bool:
    """Whether an LLM configuration keeps prompts on this machine or network."""
    if provider not in LOCAL_LLM_PROVIDERS:
        return False
    base_url = kwargs.get("base_url")
    return not base_url or is_local_url(str(base_url))
