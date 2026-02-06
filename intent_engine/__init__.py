"""Intent Engine -- prosody-aware AI for emotional intelligence.

Public API re-exports for convenient access::

    from intent_engine import IntentEngine, CloudEngine, HybridEngine, LocalEngine
    from intent_engine.training import FineTuner, MavisDataConverter, TrainingEvaluator
"""

from intent_engine.cloud_engine import CloudEngine
from intent_engine.constitutional.filter import ConstitutionalFilter
from intent_engine.engine import IntentEngine
from intent_engine.errors import IntentEngineError, LLMError, STTError, TTSError
from intent_engine.hybrid_engine import HybridEngine
from intent_engine.local_engine import LocalEngine
from intent_engine.models.audio import Audio
from intent_engine.models.decision import Decision
from intent_engine.models.response import Response
from intent_engine.models.result import Result
from intent_engine.training import FineTuner, MavisDataConverter, TrainingEvaluator

__all__ = [
    # Core engine
    "IntentEngine",
    # Deployment engines
    "CloudEngine",
    "HybridEngine",
    "LocalEngine",
    # Models
    "Result",
    "Response",
    "Audio",
    "Decision",
    # Constitutional
    "ConstitutionalFilter",
    # Training
    "FineTuner",
    "MavisDataConverter",
    "TrainingEvaluator",
    # Errors
    "IntentEngineError",
    "STTError",
    "LLMError",
    "TTSError",
]
