"""Constitutional filter -- prosody-based intent verification.

Provides the ``ConstitutionalFilter`` class that loads rules from
YAML and evaluates user intents against ``prosody_protocol.SpanFeatures``
to produce ``Decision`` objects.

Usage::

    from intent_engine.constitutional import ConstitutionalFilter

    filter = ConstitutionalFilter.from_yaml("constitutional_rules.yaml")
    decision = filter.evaluate(
        intent="delete_account",
        prosody_features=features,
        emotion="frustrated",
        emotion_confidence=0.7,  # below 0.5 the emotion counts as unknown
    )
"""

from intent_engine.constitutional.filter import ConstitutionalFilter
from intent_engine.constitutional.rules import (
    ConstitutionalRule,
    ProsodyCondition,
    Verification,
)

__all__ = [
    "ConstitutionalFilter",
    "ConstitutionalRule",
    "ProsodyCondition",
    "Verification",
]
