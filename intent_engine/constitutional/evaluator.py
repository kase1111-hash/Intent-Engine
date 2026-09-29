"""Prosody-based rule evaluation logic.

Contains the core evaluation functions that check
``prosody_protocol.SpanFeatures`` against ``ConstitutionalRule``
conditions to determine whether an action should be allowed.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Iterable, Sequence

from prosody_protocol import SpanFeatures

from intent_engine.constitutional.rules import (
    VERIFICATION_METHODS,
    ConstitutionalRule,
    ProsodyCondition,
    normalize_emotion,
    normalize_phrase,
)
from intent_engine.models.decision import Decision

logger = logging.getLogger(__name__)

# Pitch variance thresholds used when evaluating the ``pitch_variance``
# condition, in semitones of pitch movement within a word (the spread between
# the 10th and 90th percentile of the voiced F0 contour).  Semitones make the
# label independent of the speaker's register: the same intonation spans more
# Hz for a higher voice.  The bounds correspond to a +-2 and a +-4 semitone
# swing (about 4 and 8 semitones of spread); they are starting values checked
# only on synthetic speech and should be calibrated on real recordings.
PITCH_VARIANCE_THRESHOLDS = {
    "low": (0.0, 4.0),
    "normal": (4.0, 8.0),
    "high": (8.0, float("inf")),
}

# A contour needs this many voiced samples for its percentiles to mean
# something (the same limit ``prosody_protocol`` uses for its own spread).
_MIN_CONTOUR_SAMPLES = 5


def match_triggers(intent: str, triggers: Sequence[str]) -> bool:
    """Check whether an intent string matches any of the rule triggers.

    Both sides are normalised with ``normalize_phrase`` (case, ``_``/``-``
    and other separators are ignored), and a trigger matches when its
    words appear in the intent as a whole, consecutive word sequence:
    trigger ``"delete all"`` matches ``"delete_all_files"`` but neither
    ``"undelete_all"`` nor ``"delete_my_files"``.  The intent being part
    of a longer trigger is *not* a match, and a trigger without words
    never matches.  Write triggers as the shortest phrase to guard;
    inflected forms (``"payments"`` for ``"payment"``) are separate words.

    Parameters
    ----------
    intent:
        The intent label to check (e.g., ``"delete_all_files"``).
    triggers:
        Trigger keywords/phrases from a rule.

    Returns
    -------
    bool
        ``True`` if any trigger matches the intent.
    """
    if isinstance(triggers, str):  # One trigger, not a sequence of letters.
        triggers = (triggers,)
    padded_intent = f" {normalize_phrase(intent)} "
    for trigger in triggers:
        phrase = normalize_phrase(trigger)
        if phrase and f" {phrase} " in padded_intent:
            return True
    return False


def _semitones(low_hz: float, high_hz: float) -> float:
    """Return the interval from *low_hz* up to *high_hz* in semitones."""
    return 12.0 * math.log2(high_hz / low_hz)


def _span_pitch_spread(feat: SpanFeatures) -> float | None:
    """Return how far pitch moves within one span, in semitones.

    Uses the 10th-90th percentile spread of ``f0_contour``, which ignores a
    stray sample at either end, and falls back to ``f0_range`` when the
    contour is too short.  ``None`` when the span has no usable pitch.
    """
    contour = [v for v in feat.f0_contour or () if math.isfinite(v) and v > 0.0]
    if len(contour) >= _MIN_CONTOUR_SAMPLES:
        deciles = statistics.quantiles(contour, n=10)
        return _semitones(deciles[0], deciles[-1])
    if feat.f0_range is not None:
        lo, hi = feat.f0_range
        if math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo <= hi:
            return _semitones(lo, hi)
    return None


def _compute_pitch_variance(features: list[SpanFeatures]) -> float | None:
    """Compute the average pitch movement across all spans, in semitones.

    Spans without usable pitch (unvoiced, or a non-positive or non-finite
    value) are skipped; ``None`` means nothing could be measured.  The
    spans are expected to be word-sized, as the pipeline produces them: the
    spread of a longer span grows with its length.
    """
    spreads = [
        spread for feat in features if (spread := _span_pitch_spread(feat)) is not None
    ]
    return sum(spreads) / len(spreads) if spreads else None


def _compute_avg_speech_rate(features: list[SpanFeatures]) -> float | None:
    """Compute the average speech rate (syllables/second) across all spans.

    Spans without a finite rate are skipped; ``None`` means nothing
    could be measured.
    """
    rates: list[float] = []
    for feat in features:
        if feat.speech_rate is not None and math.isfinite(feat.speech_rate):
            rates.append(feat.speech_rate)
    return sum(rates) / len(rates) if rates else None


def _clean_emotion(emotion: str | None) -> str | None:
    """Return the emotion label in comparison form, or ``None`` if there is none.

    Labels are compared case-insensitively; a missing, blank or non-string
    label counts as unknown.
    """
    if not isinstance(emotion, str):
        return None
    return normalize_emotion(emotion) or None


def resolve_emotion(
    emotion: str | None,
    confidence: float | None = None,
    min_confidence: float = 0.5,
) -> str | None:
    """Return the emotion label rules should be evaluated against.

    ``None`` means the emotion is *unknown*: no label was given, or its
    ``confidence`` is below ``min_confidence`` (or not a number).  Prosody
    Protocol abstains with ``("neutral", 0.0)`` when it cannot judge, so a
    low-confidence label must not count as evidence, neither for a required
    list that contains ``"neutral"`` nor against a forbidden one.  Without a
    ``confidence`` the label is taken as given.

    Raises
    ------
    ValueError
        If ``min_confidence`` is not between 0 and 1.
    """
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(
            f"min_emotion_confidence must be between 0 and 1, got {min_confidence!r}"
        )
    label = _clean_emotion(emotion)
    if label is None:
        return None
    if confidence is not None and not confidence >= min_confidence:
        return None
    return label


def check_required_prosody(
    condition: ProsodyCondition,
    features: list[SpanFeatures],
    emotion: str | None = None,
) -> tuple[bool, str | None]:
    """Check whether prosody features satisfy a required condition.

    Parameters
    ----------
    condition:
        The required prosody condition to check.
    features:
        List of ``SpanFeatures`` from the pipeline.
    emotion:
        The detected emotion label from the IML utterance.

    Returns
    -------
    tuple[bool, str | None]
        ``(True, None)`` if all checks pass, or ``(False, reason)``
        with a human-readable explanation of what failed.
    """
    # Check emotion requirement
    if condition.emotion:
        detected = _clean_emotion(emotion)
        if detected is None:
            return (False, "Required emotion not met (emotion unknown)")
        if detected not in condition.emotion:
            return (False, "Required emotion not met (emotion not among the required labels)")

    # Check pitch variance
    if condition.pitch_variance is not None:
        variance = _compute_pitch_variance(features)
        if variance is None:
            return (
                False,
                f"Required pitch_variance='{condition.pitch_variance}' "
                "but pitch could not be measured",
            )
        thresholds = PITCH_VARIANCE_THRESHOLDS.get(condition.pitch_variance)
        if thresholds is None:
            return (False, f"Unknown pitch_variance level '{condition.pitch_variance}'")
        lo, hi = thresholds
        if not (lo <= variance < hi):
            return (
                False,
                f"Required pitch_variance='{condition.pitch_variance}' "
                f"but measured {variance:.1f} semitones",
            )

    # Check speaking rate
    if condition.speaking_rate is not None:
        avg_rate = _compute_avg_speech_rate(features)
        min_rate, max_rate = condition.speaking_rate
        if avg_rate is None:
            return (
                False,
                f"Required speaking_rate [{min_rate}, {max_rate}] "
                "but speech rate could not be measured",
            )
        if not (min_rate <= avg_rate <= max_rate):
            return (
                False,
                f"Required speaking_rate [{min_rate}, {max_rate}] "
                f"but measured {avg_rate:.1f}",
            )

    return (True, None)


def check_forbidden_prosody(
    condition: ProsodyCondition,
    features: list[SpanFeatures],
    emotion: str | None = None,
) -> tuple[bool, str | None]:
    """Check whether prosody features violate a forbidden condition.

    Parameters
    ----------
    condition:
        The forbidden prosody condition to check.
    features:
        List of ``SpanFeatures`` from the pipeline.
    emotion:
        The detected emotion label from the IML utterance.

    Returns
    -------
    tuple[bool, str | None]
        ``(True, None)`` if no forbidden conditions are violated, or
        ``(False, reason)`` if a forbidden condition was detected.
    """
    # Check forbidden emotions
    if condition.emotion:
        detected = _clean_emotion(emotion)
        if detected is not None and detected in condition.emotion:
            return (False, "Forbidden emotion detected")

    return (True, None)


def evaluate_rule(
    rule: ConstitutionalRule,
    features: list[SpanFeatures],
    emotion: str | None = None,
) -> Decision:
    """Evaluate a single constitutional rule against prosody features.

    Parameters
    ----------
    rule:
        The constitutional rule to evaluate.
    features:
        List of ``SpanFeatures`` from the pipeline.
    emotion:
        The detected emotion label from the IML utterance.

    Returns
    -------
    Decision
        The evaluation result.  ``denial_reason`` names the rule and the
        condition that failed but never repeats the emotion label.
    """
    # Check forbidden prosody first (hard block)
    if rule.forbidden_prosody:
        passed, reason = check_forbidden_prosody(
            rule.forbidden_prosody, features, emotion
        )
        if not passed:
            # Emotional data is sensitive: log the rule and outcome only.
            logger.warning("Rule '%s' denied: forbidden prosody detected", rule.name)
            logger.debug("Rule '%s': %s", rule.name, reason)
            return Decision(
                allow=False,
                requires_verification=False,
                denial_reason=f"Rule '{rule.name}': {reason}",
            )

    # Check required prosody
    if rule.required_prosody:
        passed, reason = check_required_prosody(
            rule.required_prosody, features, emotion
        )
        if not passed:
            if rule.verification:
                logger.info(
                    "Rule '%s' requires verification (%s)",
                    rule.name,
                    rule.verification.method,
                )
                logger.debug("Rule '%s': %s", rule.name, reason)
                return Decision(
                    allow=False,
                    requires_verification=True,
                    verification_method=rule.verification.method,
                    denial_reason=f"Rule '{rule.name}': {reason}",
                )
            else:
                logger.warning("Rule '%s' denied: required prosody not met", rule.name)
                logger.debug("Rule '%s': %s", rule.name, reason)
                return Decision(
                    allow=False,
                    requires_verification=False,
                    denial_reason=f"Rule '{rule.name}': {reason}",
                )

    # All checks passed
    return Decision(allow=True)


def _restrictiveness(decision: Decision) -> int:
    """Rank a decision: allow < verification (by method) < hard deny."""
    if decision.allow:
        return 0
    if not decision.requires_verification:
        return len(VERIFICATION_METHODS) + 1
    if decision.verification_method in VERIFICATION_METHODS:
        return 1 + VERIFICATION_METHODS.index(decision.verification_method)
    return len(VERIFICATION_METHODS)  # Unknown method: assume the strictest.


def most_restrictive(decisions: Iterable[Decision]) -> Decision:
    """Return the most restrictive of ``decisions`` (an allow if there are none).

    A hard deny beats ``two_factor`` verification, which beats
    ``explicit_confirmation``, which beats an allow.  The result does not
    depend on the order of ``decisions``: equally restrictive decisions
    are told apart by their ``denial_reason``, which starts with the
    rule name.
    """
    return min(
        decisions,
        key=lambda d: (-_restrictiveness(d), d.denial_reason or ""),
        default=Decision(allow=True),
    )
