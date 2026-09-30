"""Shared test fixtures and helpers for Intent Engine tests.

Provides the IML validation gate, reusable mock factories, and
sample data for integration and end-to-end tests.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from prosody_protocol import (
    IMLDocument,
    IMLValidator,
    ProsodyMapping,
    ProsodyProfile,
    Segment,
    SpanFeatures,
    Utterance,
    WordAlignment,
)

from intent_engine.engine import IntentEngine
from intent_engine.llm.base import InterpretationResult
from intent_engine.stt.base import TranscriptionResult
from intent_engine.tts.base import SynthesisResult

# ---------------------------------------------------------------------------
# IML validation gate
# ---------------------------------------------------------------------------

_validator = IMLValidator()

# ---------------------------------------------------------------------------
# Loopback traffic never meets a proxy
# ---------------------------------------------------------------------------

_PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
_LOOPBACK_HOSTS = "127.0.0.1,localhost,::1"


@pytest.fixture(autouse=True)
def _loopback_traffic_bypasses_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the tests' fake servers on ``127.0.0.1`` off any configured proxy.

    The vendor SDKs and ``httpx`` honour ``HTTP_PROXY``, ``ALL_PROXY`` and
    ``NO_PROXY`` (either case), so behind a proxy that does not exempt
    loopback every request to a fake server would go to the proxy and fail.
    No test needs a proxy, so drop them all and list the loopback hosts in
    ``NO_PROXY`` as well.  Function scope is early enough: the clients read
    the environment when a test constructs them.
    """
    for name in _PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setenv("NO_PROXY", _LOOPBACK_HOSTS)
    monkeypatch.setenv("no_proxy", _LOOPBACK_HOSTS)


def assert_valid_iml(iml_string: str) -> None:
    """Assert that an IML string passes ``prosody_protocol.IMLValidator``.

    This is the IML validation gate described in Phase 10.  Use this
    in any test that produces IML output.
    """
    result = _validator.validate(iml_string)
    assert result.valid, f"IML validation errors: {result.issues}"


@pytest.fixture()
def iml_validator() -> IMLValidator:
    """Provide a shared IMLValidator instance."""
    return _validator


# ---------------------------------------------------------------------------
# Provider SDK modules keep their identity across tests
# ---------------------------------------------------------------------------

# Top-level names of the optional SDKs the adapters import lazily.
_OPTIONAL_SDKS = (
    "anthropic",
    "assemblyai",
    "deepgram",
    "elevenlabs",
    "llama_cpp",
    "openai",
    "pyttsx3",
    "TTS",
    "whisper",
)


@pytest.fixture(autouse=True)
def _keep_loaded_sdk_modules() -> Iterator[None]:
    """Put back a provider SDK module that a test replaced or removed.

    Many adapter tests install a fake with ``sys.modules[name] = fake`` and
    finish with ``sys.modules.pop(name)``, which also removes a real SDK that
    was already imported.  The next test then imports a second copy, so patches
    made on the first one (``monkeypatch.setattr(elevenlabs, ...)``) miss the
    module the adapter sees.  New tests should use
    ``monkeypatch.setitem(sys.modules, name, fake)``; this fixture keeps the
    older ones from leaking.
    """
    loaded = {name: sys.modules[name] for name in _OPTIONAL_SDKS if name in sys.modules}
    yield
    for name, module in loaded.items():
        if sys.modules.get(name) is not module:
            sys.modules[name] = module


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------


def make_span_features(
    f0_mean: float | None = 180.0,
    intensity_mean: float | None = 65.0,
    speech_rate: float | None = 4.5,
    quality: str | None = None,
    text: str = "hello",
) -> SpanFeatures:
    """Create a SpanFeatures instance with sensible defaults."""
    return SpanFeatures(
        start_ms=0,
        end_ms=1000,
        text=text,
        f0_mean=f0_mean,
        intensity_mean=intensity_mean,
        speech_rate=speech_rate,
        quality=quality,
    )


def make_iml_document(
    emotion: str | None = None,
    confidence: float | None = None,
) -> IMLDocument:
    """Create a minimal valid IMLDocument.

    ``emotion``/``confidence`` go on its utterance, as the assembler
    writes them when it classifies an emotion.
    """
    return IMLDocument(
        utterances=(
            Utterance(children=(Segment(),), emotion=emotion, confidence=confidence),
        ),
        version="0.1.0",
    )


def make_transcription_result(
    text: str = "Hello world",
    language: str = "en",
) -> TranscriptionResult:
    """Create a mock TranscriptionResult with one timed word per token."""
    alignments = [
        WordAlignment(word=word, start_ms=300 * i, end_ms=300 * i + 250)
        for i, word in enumerate(text.split())
    ]
    return TranscriptionResult(text=text, alignments=alignments, language=language)


def make_interpretation_result(
    intent: str = "greet",
    response_text: str = "Hello! How can I help you?",
    suggested_emotion: str = "joyful",
) -> InterpretationResult:
    """Create a mock InterpretationResult."""
    return InterpretationResult(
        intent=intent,
        response_text=response_text,
        suggested_emotion=suggested_emotion,
    )


def make_synthesis_result(
    audio_data: bytes = b"RIFF fake audio data",
    format: str = "wav",
    sample_rate: int = 22050,
    duration: float = 1.5,
) -> SynthesisResult:
    """Create a mock SynthesisResult."""
    return SynthesisResult(
        audio_data=audio_data,
        format=format,
        sample_rate=sample_rate,
        duration=duration,
    )


def make_prosody_profile() -> ProsodyProfile:
    """Create a sample prosody profile for testing.

    Uses the Prosody Protocol profile vocabulary, so it validates against
    ``schemas/prosody-profile.schema.json``.
    """
    return ProsodyProfile(
        profile_version="1.0.0",
        user_id="test-user",
        description="Test profile for unit tests",
        mappings=(
            ProsodyMapping(
                pattern={"pitch": "high"},
                interpretation_emotion="joyful",
                confidence_boost=0.15,
            ),
            ProsodyMapping(
                pattern={"rate": "slow", "quality": "breathy"},
                interpretation_emotion="calm",
                confidence_boost=0.1,
            ),
        ),
    )


def make_flat_speech() -> tuple[list[WordAlignment], list[SpanFeatures]]:
    """Word alignments and per-word features of one sentence spoken monotone.

    The features carry a flat F0 contour, so a real ``IMLAssembler`` with a
    ``{"pitch_contour": "flat"}`` profile matches the utterance without any
    audio analysis.
    """
    words = ["I", "am", "fine", "thank", "you", "today."]
    alignments: list[WordAlignment] = []
    features: list[SpanFeatures] = []
    for i, word in enumerate(words):
        start = 300 * i
        alignments.append(WordAlignment(word=word, start_ms=start, end_ms=start + 250))
        features.append(
            SpanFeatures(
                start_ms=start,
                end_ms=start + 250,
                text=word,
                f0_mean=150.0,
                f0_contour=[150.0] * 25,
                intensity_mean=65.0,
                speech_rate=4.5,
                quality="modal",
            )
        )
    return alignments, features


# ---------------------------------------------------------------------------
# Mocked IntentEngine factory
# ---------------------------------------------------------------------------


def create_mocked_engine(**kwargs) -> IntentEngine:
    """Create an IntentEngine with all providers mocked.

    Returns an engine where STT, LLM, and TTS providers are MagicMock
    instances whose async methods can be configured by tests.
    """
    with patch("intent_engine.engine.create_stt_provider") as stt_f, \
         patch("intent_engine.engine.create_llm_provider") as llm_f, \
         patch("intent_engine.engine.create_tts_provider") as tts_f:
        stt_f.return_value = MagicMock()
        llm_f.return_value = MagicMock()
        tts_f.return_value = MagicMock()
        engine = IntentEngine(**kwargs)
    return engine
