"""Slack bot helper for voice channel processing.

Provides ``SlackBotHelper`` which bridges Slack voice/audio events
with the Intent Engine pipeline.  Records audio from Slack,
processes it for emotion and intent, and posts results back to a
channel.

Usage::

    from intent_engine import IntentEngine
    from intent_engine.integrations.slack import SlackBotHelper

    engine = IntentEngine()
    helper = SlackBotHelper(engine, bot_token="xoxb-...")

    # Process an uploaded audio file
    message = await helper.process_audio_file(file_url, channel_id, user_id)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Callable

from intent_engine.errors import IntentEngineError
from intent_engine.models.result import Result

logger = logging.getLogger(__name__)


class SlackBotHelper:
    """Process Slack audio through the Intent Engine pipeline.

    Downloads audio files shared in Slack channels, runs them
    through ``IntentEngine.process_voice_input()``, and formats
    results for posting back to Slack.

    Parameters
    ----------
    engine:
        An ``IntentEngine`` (or deployment variant) instance.
    bot_token:
        Slack bot token (``xoxb-...``) for API calls.
    format_callback:
        Optional callback ``(Result, user_id) -> str`` for custom
        message formatting.  Receives the pipeline result and the
        Slack user ID.
    download_func:
        Optional async callable ``(url, token) -> bytes`` for
        downloading files from Slack.  Defaults to using ``httpx``.
    """

    def __init__(
        self,
        engine: Any,
        bot_token: str | None = None,
        format_callback: Callable[[Result, str], str] | None = None,
        download_func: Callable[..., Any] | None = None,
    ) -> None:
        self._engine = engine
        self._bot_token = bot_token
        self._format_callback = format_callback
        self._download_func = download_func

    async def _download_file(self, url: str) -> bytes:
        """Download a file from Slack using the bot token."""
        if self._download_func:
            return await self._download_func(url, self._bot_token)

        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for downloading Slack files. "
                "Install it with: pip install httpx"
            )

        headers = {}
        if self._bot_token:
            headers["Authorization"] = f"Bearer {self._bot_token}"

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.content

    async def process_audio_file(
        self,
        file_url: str,
        channel_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Process an audio file from Slack and return a formatted message.

        Parameters
        ----------
        file_url:
            URL of the audio file in Slack (private download URL).
        channel_id:
            Slack channel ID for context.
        user_id:
            Slack user ID who uploaded the audio.

        Returns
        -------
        dict
            A Slack message payload with ``"channel"``, ``"text"``,
            and ``"blocks"`` fields ready for ``chat.postMessage``.
        """
        logger.info(
            "Processing Slack audio (channel=%s, user=%s)",
            channel_id,
            user_id,
        )

        audio_bytes = await self._download_file(file_url)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = await self._engine.process_voice_input(tmp_path)

            if self._format_callback and user_id:
                text = self._format_callback(result, user_id)
            else:
                text = self._format_message(result, user_id)

            return self._build_slack_message(text, channel_id, result)

        except IntentEngineError as exc:
            logger.exception("Intent Engine error processing Slack audio")
            return self._build_slack_message(
                f"Failed to process audio: {exc}",
                channel_id,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @staticmethod
    def _format_message(result: Result, user_id: str | None = None) -> str:
        """Format the default Slack message."""
        user_mention = f"<@{user_id}>" if user_id else "A user"
        return (
            f"[Detected *{result.emotion}* "
            f"({result.confidence:.0%} confidence)] "
            f"{user_mention} said: {result.text}"
        )

    @staticmethod
    def _build_slack_message(
        text: str,
        channel_id: str | None = None,
        result: Result | None = None,
    ) -> dict[str, Any]:
        """Build a Slack message payload."""
        message: dict[str, Any] = {"text": text}

        if channel_id:
            message["channel"] = channel_id

        if result:
            message["blocks"] = [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": text},
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                f"Emotion: *{result.emotion}* | "
                                f"Confidence: {result.confidence:.0%} | "
                                f"Suggested tone: *{result.suggested_tone}*"
                            ),
                        }
                    ],
                },
            ]

        return message

    async def handle_file_shared_event(
        self, event: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Handle a Slack ``file_shared`` event.

        Checks if the shared file is an audio type and processes it
        if so.  Returns ``None`` if the file is not audio.

        Parameters
        ----------
        event:
            The Slack event payload from the Events API.

        Returns
        -------
        dict or None
            A Slack message payload, or ``None`` if not audio.
        """
        file_info = event.get("file", {})
        mimetype = file_info.get("mimetype", "")

        if not mimetype.startswith("audio/"):
            return None

        file_url = file_info.get("url_private_download", "")
        channel_id = event.get("channel_id") or event.get("channel")
        user_id = event.get("user_id") or event.get("user")

        if not file_url:
            logger.warning("No download URL in file_shared event")
            return None

        return await self.process_audio_file(file_url, channel_id, user_id)
