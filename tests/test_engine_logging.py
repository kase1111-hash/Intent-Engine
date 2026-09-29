"""The engine's INFO/WARNING logs carry no personal or emotional data.

Emotional data is treated as sensitive PII: transcripts, detected emotion,
intent and the accessibility profile's ``user_id`` must not reach ordinary
application logs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from intent_engine.llm.base import InterpretationResult
from intent_engine.stt.base import TranscriptionResult
from tests.conftest import (
    create_mocked_engine,
    make_flat_speech,
    make_synthesis_result,
)

USER_ID = "patient-4471"
TRANSCRIPT = "I am fine thank you today."
SECRETS = [USER_ID, "calm", "delete_files", "fine", "thank", "x-profile"]


def _profile_file(tmp_path: Path) -> str:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "profile_version": "1.0.0",
                "user_id": USER_ID,
                "prosody_mappings": [
                    {
                        "pattern": {"pitch_contour": "flat"},
                        "interpretation": {"emotion": "calm", "confidence_boost": 0.6},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _visible(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.INFO]


def _assert_no_secrets(records: list[logging.LogRecord]) -> None:
    for record in records:
        text = record.getMessage()
        if record.exc_text:
            text += record.exc_text
        assert record.exc_info is None
        for secret in SECRETS:
            assert secret not in text, f"{secret!r} in {record.levelname} log: {text}"


def test_profile_management_does_not_log_the_user_id(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _profile_file(tmp_path)

    with caplog.at_level(logging.DEBUG, logger="intent_engine"):
        engine = create_mocked_engine(prosody_profile=path)  # load + set
        profile = engine.load_profile(path)
        engine.set_profile(profile)
        engine.clear_profile()

    assert _visible(caplog)  # the engine still says what it did
    _assert_no_secrets(_visible(caplog))


def test_a_full_turn_logs_no_transcript_emotion_or_intent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    profile_path = _profile_file(tmp_path)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF fake audio")

    with caplog.at_level(logging.DEBUG, logger="intent_engine"):
        engine = create_mocked_engine(prosody_profile=profile_path)
        alignments, features = make_flat_speech()
        engine._stt.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text=TRANSCRIPT, alignments=alignments, language="en"
            )
        )
        engine._analyzer.analyze = MagicMock(return_value=features)
        engine._analyzer.detect_pauses = MagicMock(return_value=[])
        engine._llm.interpret = AsyncMock(
            return_value=InterpretationResult(
                intent="delete_files", response_text="Are you sure?", suggested_emotion="calm"
            )
        )
        engine._tts.synthesize = AsyncMock(return_value=make_synthesis_result())

        async def turn() -> None:
            result = await engine.process_voice_input(str(audio))
            assert result.emotion == "calm"  # so the secrets really are in play
            await engine.process_voice_input(str(audio))  # cache hit
            response = await engine.generate_response(result.iml, tone=result.suggested_tone)
            assert response.intent == "delete_files"
            await engine.synthesize_speech(response.text, emotion=response.emotion)
            await engine.type_to_speech(TRANSCRIPT, emotion="calm")

        asyncio.run(turn())

    assert _visible(caplog)
    _assert_no_secrets(_visible(caplog))
