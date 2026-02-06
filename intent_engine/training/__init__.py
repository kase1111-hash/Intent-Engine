"""Training module for fine-tuning local LLMs on prosody data.

Provides :class:`FineTuner` for training models on the Prosody Protocol
dataset, :class:`MavisDataConverter` for converting Mavis game sessions
into training entries, and :class:`TrainingEvaluator` for comparing
fine-tuned models against prompt-based baselines using
``prosody_protocol.Benchmark``.
"""

from intent_engine.training.fine_tuner import FineTuner
from intent_engine.training.mavis_converter import MavisDataConverter
from intent_engine.training.evaluator import TrainingEvaluator

__all__ = [
    "FineTuner",
    "MavisDataConverter",
    "TrainingEvaluator",
]
