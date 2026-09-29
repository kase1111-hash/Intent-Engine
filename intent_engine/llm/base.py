"""Abstract LLM provider interface and shared types.

All LLM adapters implement ``LLMProvider`` and return
``InterpretationResult`` objects containing the interpreted intent,
response text, and a suggested emotion for TTS synthesis.

The reply parser lives here too, so every adapter turns a model reply
into an ``InterpretationResult`` (or an ``LLMError``) the same way.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from intent_engine.errors import LLMError

logger = logging.getLogger(__name__)

CORE_EMOTIONS: tuple[str, ...] = (
    "neutral",
    "sincere",
    "sarcastic",
    "frustrated",
    "joyful",
    "uncertain",
    "angry",
    "sad",
    "fearful",
    "surprised",
    "disgusted",
    "calm",
    "empathetic",
)
"""Prosody Protocol's core emotion vocabulary (the labels the TTS layer maps to voices)."""

_REPLY_FIELDS = ("intent", "response_text", "suggested_emotion")
_FENCE_OPENING = re.compile(r"^```[\w-]*")
_SNIPPET_LENGTH = 200


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
        the response: one of :data:`CORE_EMOTIONS` (adapters map any
        other label to ``"neutral"``).
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


def normalize_emotion(label: str) -> str:
    """Map an LLM-suggested emotion label onto :data:`CORE_EMOTIONS`.

    Case and surrounding whitespace are ignored.  A label outside the
    vocabulary becomes ``"neutral"`` (the TTS layer would fall back to it
    anyway) and a warning is logged.  The label itself is not logged above
    DEBUG level because emotional data is sensitive.
    """
    label = label.strip().lower()
    if label in CORE_EMOTIONS:
        return label
    logger.warning("LLM suggested an emotion outside the core vocabulary; using 'neutral'")
    logger.debug("Unrecognised suggested emotion: %r", label)
    return "neutral"


def _snippet(raw: str) -> str:
    return raw[:_SNIPPET_LENGTH]


def _reply_object(raw: str, source: str) -> dict[str, Any]:
    """Find the JSON object in a model reply.

    Models sometimes wrap the object in a markdown fence or surround it
    with a sentence of prose, so text that is not a bare object is
    searched for the first ``{`` that starts a valid JSON object.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = _FENCE_OPENING.sub("", text).removesuffix("```").strip()

    # RecursionError: json gives up on very deeply nested (degenerate) replies.
    try:
        whole = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        decoder = json.JSONDecoder()
        start = text.find("{")
        while start != -1:
            try:
                # Starting at "{", a successful decode is always an object.
                found: dict[str, Any] = decoder.raw_decode(text, start)[0]
            except (json.JSONDecodeError, RecursionError):
                start = text.find("{", start + 1)
            else:
                return found
        raise LLMError(f"{source} returned non-JSON response: {_snippet(raw)}") from None

    if not isinstance(whole, dict):
        raise LLMError(f"{source} returned JSON that is not an object: {_snippet(raw)}")
    return whole


def parse_interpretation(raw: str, source: str) -> InterpretationResult:
    """Parse a model reply into an :class:`InterpretationResult`.

    Parameters
    ----------
    raw:
        The reply text.  It should be the JSON object described in the
        system prompt, but a markdown fence or a sentence around it is
        tolerated.
    source:
        Name of the provider, used in error messages (e.g. ``"Claude"``).

    Returns
    -------
    InterpretationResult
        ``intent`` and ``response_text`` with surrounding whitespace
        removed and ``suggested_emotion`` normalised by
        :func:`normalize_emotion`.

    Raises
    ------
    intent_engine.errors.LLMError
        If the reply is not a JSON object, or a required field is
        missing, is not a string, or is blank.
    """
    obj = _reply_object(raw, source)

    fields: dict[str, str] = {}
    for name in _REPLY_FIELDS:
        value = obj.get(name)
        if not isinstance(value, str) or not value.strip():
            raise LLMError(f"{source} response has no usable {name!r} string: {_snippet(raw)}")
        fields[name] = value.strip()

    return InterpretationResult(
        intent=fields["intent"],
        response_text=fields["response_text"],
        suggested_emotion=normalize_emotion(fields["suggested_emotion"]),
    )


def _field(obj: Any, name: str) -> Any:
    """Read ``name`` from a dict (llama.cpp) or an SDK object (OpenAI client)."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def chat_completion_text(response: Any, source: str) -> str:
    """Return the assistant text of an OpenAI-style chat completion.

    Works on both the OpenAI SDK's response object and the plain dict that
    ``llama-cpp-python`` returns.

    Raises
    ------
    intent_engine.errors.LLMError
        If the response has no choices or the message has no text (for
        example because the model refused; the refusal is quoted).
    """
    choices = _field(response, "choices")
    if not choices:
        raise LLMError(f"{source} returned no choices")
    message = _field(choices[0], "message")
    content = _field(message, "content")
    if isinstance(content, str) and content.strip():
        return content
    refusal = _field(message, "refusal")
    if isinstance(refusal, str) and refusal:
        raise LLMError(f"{source} refused to answer: {_snippet(refusal)}")
    raise LLMError(f"{source} returned no text")
