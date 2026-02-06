"""Result dataclass -- output of IntentEngine.process_voice_input()."""

from __future__ import annotations

from dataclasses import dataclass

from prosody_protocol import IMLDocument, SpanFeatures


@dataclass(frozen=True)
class Result:
    """Output of ``IntentEngine.process_voice_input()``.

    Wraps the transcription text, detected emotion, IML document (from
    ``prosody_protocol``), and extracted prosodic features into a single
    immutable object.
    """

    text: str
    """Plain text transcription."""

    emotion: str
    """Primary detected emotion (from Prosody Protocol core vocabulary)."""

    confidence: float
    """Emotion classification confidence, 0.0 to 1.0."""

    iml: str
    """Serialized IML markup string."""

    iml_document: IMLDocument
    """Parsed IML document (from ``prosody_protocol``)."""

    suggested_tone: str
    """Recommended emotional tone for the response."""

    prosody_features: list[SpanFeatures]
    """Per-span prosodic features extracted from audio (from ``prosody_protocol``)."""

    intent: str | None = None
    """Parsed user intent, or ``None`` if not yet interpreted."""
