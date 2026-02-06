"""Prosody-based rule evaluation logic.

Contains the core evaluation functions that check
``prosody_protocol.SpanFeatures`` against ``ConstitutionalRule``
conditions to determine whether an action should be allowed.
"""

from __future__ import annotations

import logging

from prosody_protocol import SpanFeatures

from intent_engine.constitutional.rules import ConstitutionalRule, ProsodyCondition
from intent_engine.models.decision import Decision

logger = logging.getLogger(__name__)

# Pitch variance thresholds (in Hz) used when evaluating
# the ``pitch_variance`` condition.
PITCH_VARIANCE_THRESHOLDS = {
    "low": (0.0, 30.0),
    "normal": (30.0, 80.0),
    "high": (80.0, float("inf")),
}


def match_triggers(intent: str, triggers: list[str]) -> bool:
    """Check whether an intent string matches any of the rule triggers.

    Matching is case-insensitive. A trigger matches if it appears as a
    substring of the intent or the intent appears as a substring of the
    trigger.

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
    intent_lower = intent.lower()
    for trigger in triggers:
        trigger_lower = trigger.lower()
        if trigger_lower in intent_lower or intent_lower in trigger_lower:
            return True
    return False


def _compute_pitch_variance(features: list[SpanFeatures]) -> float | None:
    """Compute the average pitch range across all spans.

    Uses ``f0_range`` (min, max) when available to calculate
    the pitch spread for each span, then averages them.
    """
    variances: list[float] = []
    for feat in features:
        if feat.f0_range is not None:
            lo, hi = feat.f0_range
            variances.append(hi - lo)
    return sum(variances) / len(variances) if variances else None


def _compute_avg_speech_rate(features: list[SpanFeatures]) -> float | None:
    """Compute the average speech rate across all spans."""
    rates: list[float] = []
    for feat in features:
        if feat.speech_rate is not None:
            rates.append(feat.speech_rate)
    return sum(rates) / len(rates) if rates else None


def _get_dominant_emotion(
    features: list[SpanFeatures], emotion: str | None
) -> str | None:
    """Return the emotion label for evaluation.

    If an explicit emotion is provided (from the IML utterance), use
    that.  Otherwise return ``None`` (no emotion data available).
    """
    return emotion


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
        detected = _get_dominant_emotion(features, emotion)
        if detected is None or detected not in condition.emotion:
            return (
                False,
                f"Required emotion {condition.emotion} but detected '{detected}'",
            )

    # Check pitch variance
    if condition.pitch_variance is not None:
        variance = _compute_pitch_variance(features)
        if variance is not None:
            thresholds = PITCH_VARIANCE_THRESHOLDS.get(condition.pitch_variance)
            if thresholds:
                lo, hi = thresholds
                if not (lo <= variance < hi):
                    return (
                        False,
                        f"Required pitch_variance='{condition.pitch_variance}' "
                        f"but measured {variance:.1f}Hz",
                    )

    # Check speaking rate
    if condition.speaking_rate is not None:
        avg_rate = _compute_avg_speech_rate(features)
        if avg_rate is not None:
            min_rate, max_rate = condition.speaking_rate
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
        detected = _get_dominant_emotion(features, emotion)
        if detected is not None and detected in condition.emotion:
            return (
                False,
                f"Forbidden emotion '{detected}' detected",
            )

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
        The evaluation result.
    """
    # Check forbidden prosody first (hard block)
    if rule.forbidden_prosody:
        passed, reason = check_forbidden_prosody(
            rule.forbidden_prosody, features, emotion
        )
        if not passed:
            logger.warning(
                "Rule '%s' denied: forbidden prosody detected -- %s",
                rule.name,
                reason,
            )
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
                    "Rule '%s' requires verification: %s",
                    rule.name,
                    reason,
                )
                return Decision(
                    allow=False,
                    requires_verification=True,
                    verification_method=rule.verification.method,
                    denial_reason=f"Rule '{rule.name}': {reason}",
                )
            else:
                logger.warning(
                    "Rule '%s' denied: required prosody not met -- %s",
                    rule.name,
                    reason,
                )
                return Decision(
                    allow=False,
                    requires_verification=False,
                    denial_reason=f"Rule '{rule.name}': {reason}",
                )

    # All checks passed
    return Decision(allow=True)
