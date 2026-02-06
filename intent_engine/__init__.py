"""Intent Engine -- prosody-aware AI for emotional intelligence.

Public API re-exports for convenient access::

    from intent_engine import IntentEngine, ConstitutionalFilter
"""

from intent_engine.constitutional.filter import ConstitutionalFilter
from intent_engine.engine import IntentEngine
from intent_engine.errors import IntentEngineError, LLMError, STTError, TTSError
from intent_engine.models.audio import Audio
from intent_engine.models.decision import Decision
from intent_engine.models.response import Response
from intent_engine.models.result import Result

__all__ = [
    # Core engine
    "IntentEngine",
    # Models
    "Result",
    "Response",
    "Audio",
    "Decision",
    # Constitutional
    "ConstitutionalFilter",
    # Errors
    "IntentEngineError",
    "STTError",
    "LLMError",
    "TTSError",
]
