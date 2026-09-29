"""Slack bot helper for voice channel processing.

Provides ``SlackBotHelper`` which bridges Slack voice/audio events
with the Intent Engine pipeline.  Records audio from Slack,
processes it for emotion and intent, and posts results back to a
channel.

Usage (from the repository root)::

    from intent_engine import IntentEngine
    from examples.integrations.slack_bot import SlackBotHelper

    engine = IntentEngine()
    helper = SlackBotHelper(engine, bot_token="xoxb-...")

    # In your Events API endpoint, first check the request really came
    # from Slack (raw body, not re-serialized JSON):
    if not SlackBotHelper.verify_signature(raw_body, request_headers, signing_secret):
        ...  # respond 401

    # Process an uploaded audio file
    message = await helper.process_audio_file(file_url, channel_id, user_id)
    client.chat_postMessage(**message)   # slack_sdk WebClient

Emotion is sensitive personal data: this posts a named user's transcript
and inferred emotion into a shared channel.  Tell the people in the channel,
and only run it where they have agreed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from intent_engine.models.result import Result

from ._common import (
    DEFAULT_MAX_DOWNLOAD_BYTES,
    download_media,
    emotion_reported,
    temp_audio_file,
)

logger = logging.getLogger(__name__)

SLACK_HOSTS = ("slack.com",)
"""Hosts (and their subdomains) files are downloaded from by default."""

_MAX_TEXT_CHARS = 3000
"""Slack rejects a section block whose text is longer than this."""


def _escape_mrkdwn(text: str) -> str:
    """Escape the three characters Slack treats as control characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _truncate(text: str, limit: int = _MAX_TEXT_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


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
        downloading files from Slack.  Defaults to a size-capped
        ``httpx`` download that only contacts ``https://*.slack.com``, so
        the bot token can never be sent to another host; a custom function
        is responsible for its own restrictions.
    max_download_bytes:
        Largest file the default downloader accepts.
    allowed_hosts:
        Hosts (and their subdomains) the default downloader may fetch from.
    """

    def __init__(
        self,
        engine: Any,
        bot_token: str | None = None,
        format_callback: Callable[[Result, str], str] | None = None,
        download_func: Callable[..., Any] | None = None,
        *,
        max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
        allowed_hosts: Iterable[str] = SLACK_HOSTS,
    ) -> None:
        self._engine = engine
        self._bot_token = bot_token
        self._format_callback = format_callback
        self._download_func = download_func
        self._max_download_bytes = max_download_bytes
        self._allowed_hosts = tuple(allowed_hosts)

    async def _download_file(self, url: str) -> bytes:
        """Download a file from Slack using the bot token."""
        if self._download_func:
            result: bytes = await self._download_func(url, self._bot_token)
            return result

        headers = {}
        if self._bot_token:
            headers["Authorization"] = f"Bearer {self._bot_token}"

        return await download_media(
            url,
            allowed_hosts=self._allowed_hosts,
            max_bytes=self._max_download_bytes,
            headers=headers,
        )

    @staticmethod
    def verify_signature(
        body: str | bytes,
        headers: Mapping[str, str],
        signing_secret: str,
    ) -> bool:
        """Check that an Events API request was signed by Slack.

        Call this on every request before trusting its payload, in
        particular before passing an event to :meth:`handle_file_shared_event`.

        Parameters
        ----------
        body:
            The raw request body, exactly as received.
        headers:
            The request headers (``X-Slack-Request-Timestamp`` and
            ``X-Slack-Signature`` are read; names are case-insensitive).
        signing_secret:
            Your app's signing secret.

        Returns
        -------
        bool
            ``True`` if the signature matches and the request is recent.
        """
        try:
            from slack_sdk.signature import SignatureVerifier
        except ImportError as exc:
            raise ImportError(
                "slack_sdk is required for signature verification. "
                "Install it with: pip install slack_sdk"
            ) from exc

        verifier = SignatureVerifier(signing_secret)
        try:
            return verifier.is_valid_request(body, headers)
        except ValueError:  # non-numeric timestamp header
            return False

    async def process_audio_file(
        self,
        file_url: str,
        channel_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Process an audio file from Slack and return a formatted message.

        Never raises for a bad or undecodable file or a pipeline failure:
        the returned message then says so generically, and the details are
        logged, not posted to the channel.

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
            and (when an emotion was reported) ``"blocks"`` fields ready
            for ``chat.postMessage``.
        """
        logger.debug(
            "Processing Slack audio (channel=%s, user=%s)",
            channel_id,
            user_id,
        )

        try:
            audio_bytes = await self._download_file(file_url)

            async with temp_audio_file(audio_bytes) as tmp_path:
                result = await self._engine.process_voice_input(tmp_path)

            if self._format_callback and user_id:
                text = self._format_callback(result, user_id)
            else:
                text = self._format_message(result, user_id)

            return self._build_slack_message(text, channel_id, result)

        except Exception:
            # One bad clip must not crash the event handler, and the
            # channel must not see provider or filesystem details.
            logger.exception("Error processing Slack audio")
            return self._build_slack_message(
                "Failed to process audio.",
                channel_id,
            )

    @staticmethod
    def _format_message(result: Result, user_id: str | None = None) -> str:
        """Format the default Slack message.

        The emotion is only mentioned when the engine reported one.
        """
        user_mention = f"<@{user_id}>" if user_id else "A user"
        detected = ""
        if emotion_reported(result):
            detected = f"[Detected *{result.emotion}* ({result.confidence:.0%} confidence)] "
        return f"{detected}{user_mention} said: {_escape_mrkdwn(result.text)}"

    @staticmethod
    def _build_slack_message(
        text: str,
        channel_id: str | None = None,
        result: Result | None = None,
    ) -> dict[str, Any]:
        """Build a Slack message payload.

        Text is cut to Slack's 3000 character limit for section blocks.
        Blocks are only added when there is an emotion to show.
        """
        text = _truncate(text)
        message: dict[str, Any] = {"text": text}

        if channel_id:
            message["channel"] = channel_id

        if result is not None and emotion_reported(result):
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

        Only call this for events from a request whose signature was
        verified with :meth:`verify_signature`.  The event must carry the
        file's ``mimetype`` and ``url_private_download``; if the payload you
        receive holds only a file ID, look the file up with ``files.info``
        first and pass the enriched event.

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
