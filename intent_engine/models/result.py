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
    """Primary detected emotion, as reported in :attr:`iml_document`.

    The default classifier can report ``calm``, ``sad``, ``angry``,
    ``joyful`` and ``fearful``; a prosody profile can map to any label.
    ``"neutral"`` with :attr:`confidence` ``0.0`` means no emotion was
    reported: the classifier was not confident, or (without calibration
    speech) the recording had too few utterances to tell the speaker's
    usual level, which single sentences never do.
    """

    confidence: float
    """Emotion classification confidence, 0.0 to 1.0 (0.0 when none was reported)."""

    iml: str
    """Serialized IML markup string."""

    iml_document: IMLDocument
    """Parsed IML document (from ``prosody_protocol``)."""

    suggested_tone: str
    """Tone of the user's voice worth acting on: :attr:`emotion` when
    :attr:`confidence` is at least 0.5, otherwise ``"neutral"``.

    Pass it to ``IntentEngine.generate_response(tone=...)`` as a hint. It
    describes the user, not the reply: the tone to speak the reply in is
    ``Response.emotion``, which the LLM chooses (an angry caller may need a
    calm reply).  ``"neutral"`` means no reading (the engine never reports a
    measured neutral), so ``generate_response`` sends no hint for it."""

    prosody_features: list[SpanFeatures]
    """Per-span prosodic features extracted from audio (from ``prosody_protocol``)."""

    intent: str | None = None
    """Parsed user intent, or ``None`` if not yet interpreted.

    ``process_voice_input()`` runs before the LLM and leaves this ``None``;
    the intent the LLM parsed is ``Response.intent``.
    """
