"""An unusable model reply must not leak its content through the error or the logs.

A reply can echo what the user said, the intent label and the emotion, all of
which are sensitive, and an exception message ends up in application logs and
tracebacks.  The ``LLMError`` therefore says what was wrong and how long the
reply was, and the start of the reply is only logged at DEBUG.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from intent_engine.errors import LLMError
from intent_engine.llm.base import chat_completion_text, parse_interpretation
from tests.llm.test_adapter_replies import DRIVERS, GOOD

SECRET = "chemo bills for Mrs. Alvarez"


def _reply(**overrides: object) -> str:
    return json.dumps({**GOOD, "response_text": f"I hear you about the {SECRET}", **overrides})


def _without(key: str) -> str:
    return json.dumps({k: v for k, v in GOOD.items() if k != key} | {"note": SECRET})


UNUSABLE = {
    "not json": f"I would rather not discuss the {SECRET}",
    "not an object": json.dumps([SECRET]),
    "missing field": _without("suggested_emotion"),
    "blank field": _reply(suggested_emotion="  "),
    "wrong type": _reply(intent=[SECRET]),
    "truncated": _reply()[:-8],
}


@pytest.mark.parametrize("reply", list(UNUSABLE.values()), ids=list(UNUSABLE))
def test_error_message_does_not_quote_the_reply(reply: str) -> None:
    with pytest.raises(LLMError) as info:
        parse_interpretation(reply, "Widget")

    message = str(info.value)
    assert "Widget" in message
    assert f"reply length {len(reply)}" in message
    assert SECRET not in message
    assert "Alvarez" not in message
    assert GOOD["intent"] not in message


@pytest.mark.parametrize("reply", list(UNUSABLE.values()), ids=list(UNUSABLE))
def test_the_start_of_the_reply_is_only_logged_at_debug(
    reply: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG, logger="intent_engine.llm.base"), pytest.raises(LLMError):
        parse_interpretation(reply, "Widget")

    assert all(SECRET not in r.getMessage() for r in caplog.records if r.levelno >= logging.INFO)
    assert any(reply[:40] in r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)


def test_the_logged_start_of_the_reply_is_bounded(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="intent_engine.llm.base"), pytest.raises(LLMError):
        parse_interpretation("x" * 5000, "Widget")

    assert all(len(r.getMessage()) < 400 for r in caplog.records)


def test_a_missing_field_is_still_named() -> None:
    with pytest.raises(LLMError, match="'suggested_emotion'"):
        parse_interpretation(_without("suggested_emotion"), "Widget")


def test_a_refusal_is_reported_without_quoting_it(caplog: pytest.LogCaptureFixture) -> None:
    refusal = f"I can't help with the {SECRET}"
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, refusal=refusal))]
    )

    with (
        caplog.at_level(logging.DEBUG, logger="intent_engine.llm.base"),
        pytest.raises(LLMError, match="refused to answer") as info,
    ):
        chat_completion_text(response, "Widget")

    assert SECRET not in str(info.value)
    assert f"refusal length {len(refusal)}" in str(info.value)
    assert all(SECRET not in r.getMessage() for r in caplog.records if r.levelno >= logging.INFO)
    assert any(SECRET in r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)


@pytest.mark.parametrize("provider", list(DRIVERS))
@pytest.mark.parametrize("reply", list(UNUSABLE.values()), ids=list(UNUSABLE))
def test_no_adapter_leaks_the_reply(
    provider: str,
    reply: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    driver: Callable[[pytest.MonkeyPatch, str], object] = DRIVERS[provider]

    with caplog.at_level(logging.INFO), pytest.raises(LLMError) as info:
        driver(monkeypatch, reply)

    assert SECRET not in str(info.value)
    assert SECRET not in caplog.text


def test_the_errors_survive_being_wrapped_and_logged_with_a_traceback(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What an application logs for a failed turn (``exc_info``) carries no reply text."""
    driver = DRIVERS["claude"]
    logger = logging.getLogger("app")

    with caplog.at_level(logging.INFO):
        try:
            driver(monkeypatch, UNUSABLE["missing field"])
        except LLMError:
            logger.warning("turn failed", exc_info=True)

    assert caplog.records
    assert SECRET not in caplog.text
