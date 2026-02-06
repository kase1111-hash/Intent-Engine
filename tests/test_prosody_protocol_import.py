"""Verify that prosody_protocol is importable and provides expected symbols.

This test validates Phase 0 deliverable: the prosody-protocol dependency
is correctly installed and all expected public API symbols are accessible.
"""

from __future__ import annotations


def test_core_imports() -> None:
    from prosody_protocol import IMLParser, IMLValidator

    assert IMLParser is not None
    assert IMLValidator is not None


def test_model_imports() -> None:
    from prosody_protocol import (
        Emphasis,
        IMLDocument,
        Pause,
        Prosody,
        Segment,
        Utterance,
    )

    assert IMLDocument is not None
    assert Utterance is not None
    assert Prosody is not None
    assert Pause is not None
    assert Emphasis is not None
    assert Segment is not None


def test_analysis_imports() -> None:
    from prosody_protocol import (
        PauseInterval,
        ProsodyAnalyzer,
        SpanFeatures,
        WordAlignment,
    )

    assert ProsodyAnalyzer is not None
    assert SpanFeatures is not None
    assert WordAlignment is not None
    assert PauseInterval is not None


def test_conversion_imports() -> None:
    from prosody_protocol import IMLAssembler, IMLToSSML

    assert IMLAssembler is not None
    assert IMLToSSML is not None


def test_emotion_imports() -> None:
    from prosody_protocol import EmotionClassifier, RuleBasedEmotionClassifier

    assert EmotionClassifier is not None
    assert RuleBasedEmotionClassifier is not None


def test_profile_imports() -> None:
    from prosody_protocol import ProfileApplier, ProfileLoader, ProsodyProfile

    assert ProfileLoader is not None
    assert ProfileApplier is not None
    assert ProsodyProfile is not None


def test_exception_imports() -> None:
    from prosody_protocol import (
        AudioProcessingError,
        ConversionError,
        IMLParseError,
        IMLValidationError,
        ProfileError,
        ProsodyProtocolError,
    )

    assert issubclass(IMLParseError, ProsodyProtocolError)
    assert issubclass(IMLValidationError, ProsodyProtocolError)
    assert issubclass(ProfileError, ProsodyProtocolError)
    assert issubclass(AudioProcessingError, ProsodyProtocolError)
    assert issubclass(ConversionError, ProsodyProtocolError)


def test_intent_engine_top_level_imports() -> None:
    from intent_engine import Audio, Decision, Response, Result

    assert Result is not None
    assert Response is not None
    assert Audio is not None
    assert Decision is not None
