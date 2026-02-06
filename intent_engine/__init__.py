"""Intent Engine -- prosody-aware AI for emotional intelligence.

Public API re-exports for convenient access::

    from intent_engine import IntentEngine, ConstitutionalFilter
"""

from intent_engine.models.audio import Audio
from intent_engine.models.decision import Decision
from intent_engine.models.response import Response
from intent_engine.models.result import Result

__all__ = [
    # Core engine (Phase 6+)
    # "IntentEngine",
    # "CloudEngine",
    # "HybridEngine",
    # "LocalEngine",
    # Models (Phase 1)
    "Result",
    "Response",
    "Audio",
    "Decision",
    # Constitutional (Phase 5)
    # "ConstitutionalFilter",
]
