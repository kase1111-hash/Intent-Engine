"""ConstitutionalFilter -- prosody-based intent verification.

Loads constitutional rules from YAML and evaluates user intents
against prosodic features to determine whether an action should
be allowed, denied, or require additional verification.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

from prosody_protocol import SpanFeatures

from intent_engine.constitutional.evaluator import (
    evaluate_rule,
    match_triggers,
    most_restrictive,
    resolve_emotion,
)
from intent_engine.constitutional.rules import ConstitutionalRule, parse_rules_yaml
from intent_engine.models.decision import Decision

logger = logging.getLogger(__name__)


class ConstitutionalFilter:
    """Safety filter that verifies intent using prosodic features.

    Rules are loaded from a YAML file (see ``from_yaml()``) or
    constructed directly from a list of ``ConstitutionalRule`` objects.

    Usage::

        filter = ConstitutionalFilter.from_yaml("rules.yaml")
        decision = filter.evaluate(
            intent="delete_account",
            prosody_features=features,
            emotion="frustrated",
        )
        if not decision.allow:
            print(decision.denial_reason)
    """

    def __init__(self, rules: Iterable[ConstitutionalRule]) -> None:
        if isinstance(rules, (str, bytes, Mapping)):
            raise TypeError(
                "rules must be an iterable of ConstitutionalRule objects; "
                "use ConstitutionalFilter.from_yaml() to load rules from a file"
            )
        self._rules = list(rules)
        for rule in self._rules:
            if not isinstance(rule, ConstitutionalRule):
                raise TypeError(f"rules must be ConstitutionalRule objects, got {rule!r}")
        logger.info(
            "ConstitutionalFilter initialized with %d rules", len(self._rules)
        )
        if not self._rules:
            logger.warning("ConstitutionalFilter has no rules: every action will be allowed")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ConstitutionalFilter:
        """Create a filter from a YAML rules file.

        Parameters
        ----------
        path:
            Path to the YAML file containing rule definitions.

        Returns
        -------
        ConstitutionalFilter
            A filter initialized with the parsed rules.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        ValueError
            If the file cannot be read or does not follow the rule schema
            (see ``parse_rules_yaml``).
        """
        rules = parse_rules_yaml(path)
        return cls(rules)

    @property
    def rules(self) -> list[ConstitutionalRule]:
        """Return a copy of the loaded rules."""
        return list(self._rules)

    def evaluate(
        self,
        intent: str,
        prosody_features: list[SpanFeatures],
        emotion: str | None = None,
        context: dict[str, object] | None = None,
        *,
        emotion_confidence: float | None = None,
        min_emotion_confidence: float = 0.5,
    ) -> Decision:
        """Evaluate an intent against constitutional rules.

        Implements the decision logic:

        1. If no rules match the intent, allow the action.
        2. If rules match and prosody passes all checks, allow.
        3. If rules match and prosody fails with verification
           defined, deny with ``requires_verification=True``.
        4. If rules match and prosody fails with no verification,
           deny outright.
        5. Every matching rule is evaluated and the most restrictive
           decision wins: a hard deny beats ``two_factor``, which beats
           ``explicit_confirmation``, which beats an allow, whatever the
           order of the rules.

        Parameters
        ----------
        intent:
            The parsed user intent label (e.g., ``"delete_account"``).
        prosody_features:
            List of ``SpanFeatures`` from the pipeline.
        emotion:
            The detected emotion label from the IML utterance, compared
            case-insensitively.  Without ``emotion_confidence`` the label is
            taken at face value: passing only ``Result.emotion`` reads the
            ``("neutral", 0.0)`` of "no emotion reported" as a measured
            neutral speaker, which passes a rule that lists ``neutral`` as
            acceptable.  When gating on a ``Result``, use
            ``IntentEngine.evaluate_result``.
        context:
            Optional context dict (e.g., ``{"user_id": "...", "session_risk": "low"}``).
            Accepted for forward compatibility; rules do not use it yet.
        emotion_confidence:
            Confidence (0-1) of ``emotion``, if the source reports one.
            When it is below ``min_emotion_confidence`` the emotion is
            treated as unknown, so an abstention such as Prosody
            Protocol's ``("neutral", 0.0)`` is not evidence of anything.
        min_emotion_confidence:
            Lowest confidence at which ``emotion`` is believed (default
            0.5, Prosody Protocol's own reporting threshold).

        Returns
        -------
        Decision
            The evaluation result.  ``denial_reason`` names the rule and the
            condition that failed but never repeats the emotion label, and
            the filter logs only rule names and outcomes (emotional data is
            sensitive).
        """
        if not isinstance(intent, str):
            raise TypeError(f"intent must be a string, got {intent!r}")
        emotion = resolve_emotion(emotion, emotion_confidence, min_emotion_confidence)

        matching_rules = [
            rule for rule in self._rules
            if match_triggers(intent, rule.triggers)
        ]

        if not matching_rules:
            logger.debug("No rules match intent '%s' -- allowing", intent)
            return Decision(allow=True)

        logger.debug(
            "Intent '%s' matched %d rules: %s",
            intent,
            len(matching_rules),
            [r.name for r in matching_rules],
        )

        # Evaluate every matching rule; the most restrictive decision wins,
        # so the outcome does not depend on the order of the rules.
        return most_restrictive(
            evaluate_rule(rule, prosody_features, emotion) for rule in matching_rules
        )
