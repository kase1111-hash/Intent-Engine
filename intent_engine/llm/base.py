"""Abstract LLM provider interface and shared types.

All LLM adapters implement ``LLMProvider`` and return
``InterpretationResult`` objects containing the interpreted intent,
response text, and a suggested emotion for TTS synthesis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class InterpretationResult:
    """Output of an LLM provider's ``interpret()`` call.

    Attributes
    ----------
    intent:
        Parsed user intent (e.g., ``"request_cancellation"``,
        ``"express_frustration"``).
    response_text:
        Generated response text to send back to the user.
    suggested_emotion:
        Emotion label the TTS layer should use when synthesizing
        the response (from Prosody Protocol's core vocabulary).
    """

    intent: str
    response_text: str
    suggested_emotion: str


class LLMProvider(ABC):
    """Abstract base class for LLM provider adapters.

    Subclasses must implement :meth:`interpret` which receives
    IML-annotated text (serialized by ``prosody_protocol.IMLParser``)
    and produces an intent interpretation with a suggested response.

    The orchestrator (``IntentEngine``) feeds the IML string from
    ``prosody_protocol.IMLAssembler`` into this method.
    """

    @abstractmethod
    async def interpret(
        self, iml_input: str, context: str | None = None
    ) -> InterpretationResult:
        """Interpret IML-annotated input and generate a response.

        Parameters
        ----------
        iml_input:
            Serialized IML markup string containing prosodic annotations
            (e.g., ``<utterance emotion="frustrated" confidence="0.85">``).
        context:
            Optional conversation context or system instructions to
            guide the interpretation.

        Returns
        -------
        InterpretationResult
            The parsed intent, response text, and suggested emotion.

        Raises
        ------
        intent_engine.errors.LLMError
            If the LLM call fails.
        """
        ...
