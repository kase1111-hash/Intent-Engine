"""ConstitutionalFilter -- prosody-based intent verification.

Loads constitutional rules from YAML and evaluates user intents
against prosodic features to determine whether an action should
be allowed, denied, or require additional verification.
"""

from __future__ import annotations

import logging
from pathlib import Path

from prosody_protocol import SpanFeatures

from intent_engine.constitutional.evaluator import evaluate_rule, match_triggers
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

    def __init__(self, rules: list[ConstitutionalRule]) -> None:
        self._rules = list(rules)
        logger.info(
            "ConstitutionalFilter initialized with %d rules", len(self._rules)
        )

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
    ) -> Decision:
        """Evaluate an intent against constitutional rules.

        Implements the decision logic:

        1. If no rules match the intent, allow the action.
        2. If rules match and prosody passes all checks, allow.
        3. If rules match and prosody fails with verification
           defined, deny with ``requires_verification=True``.
        4. If rules match and prosody fails with no verification,
           deny outright.

        Parameters
        ----------
        intent:
            The parsed user intent label (e.g., ``"delete_account"``).
        prosody_features:
            List of ``SpanFeatures`` from the pipeline.
        emotion:
            The detected emotion label from the IML utterance.
        context:
            Optional context dict (e.g., ``{"user_id": "...", "session_risk": "low"}``).

        Returns
        -------
        Decision
            The evaluation result.
        """
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

        # Evaluate all matching rules; the most restrictive decision wins
        for rule in matching_rules:
            decision = evaluate_rule(rule, prosody_features, emotion)
            if not decision.allow:
                return decision

        return Decision(allow=True)
