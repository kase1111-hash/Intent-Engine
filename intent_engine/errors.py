"""Intent Engine error hierarchy.

All provider-specific exceptions inherit from ``IntentEngineError``
so callers can catch a single base class for Intent Engine failures
while still handling specific error types when needed.

Prosody Protocol exceptions (``AudioProcessingError``,
``IMLParseError``, etc.) are re-raised as-is -- they already have
clear error messages and are documented in the Prosody Protocol SDK.
"""

from __future__ import annotations


class IntentEngineError(Exception):
    """Base exception for all Intent Engine errors."""


class STTError(IntentEngineError):
    """Raised when speech-to-text transcription fails."""


class LLMError(IntentEngineError):
    """Raised when LLM intent interpretation fails."""


class TTSError(IntentEngineError):
    """Raised when text-to-speech synthesis fails."""
