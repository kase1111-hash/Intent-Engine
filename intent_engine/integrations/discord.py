"""Discord bot helper for voice channel processing.

Provides ``DiscordBotHelper`` which bridges Discord voice events
with the Intent Engine pipeline.  Processes audio from voice
channels or uploaded files, detects emotion and intent, and posts
results back to a text channel.

Usage::

    from intent_engine import IntentEngine
    from intent_engine.integrations.discord import DiscordBotHelper

    engine = IntentEngine()
    helper = DiscordBotHelper(engine)

    @bot.event
    async def on_message(message):
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("audio/"):
                result = await helper.process_audio_attachment(attachment, message.channel)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Callable

from intent_engine.errors import IntentEngineError
from intent_engine.models.result import Result

logger = logging.getLogger(__name__)


class DiscordBotHelper:
    """Process Discord audio through the Intent Engine pipeline.

    Downloads audio from Discord attachments or voice recordings,
    processes through ``IntentEngine.process_voice_input()``, and
    returns formatted messages with emotion context for text channels.

    Parameters
    ----------
    engine:
        An ``IntentEngine`` (or deployment variant) instance.
    format_callback:
        Optional callback ``(Result, user_id) -> str`` for custom
        message formatting.
    download_func:
        Optional async callable ``(url) -> bytes`` for downloading
        audio.  Defaults to using ``httpx``.
    """

    def __init__(
        self,
        engine: Any,
        format_callback: Callable[[Result, str], str] | None = None,
        download_func: Callable[..., Any] | None = None,
    ) -> None:
        self._engine = engine
        self._format_callback = format_callback
        self._download_func = download_func

    async def _download_audio(self, url: str) -> bytes:
        """Download audio bytes from a URL."""
        if self._download_func:
            return await self._download_func(url)

        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for downloading Discord attachments. "
                "Install it with: pip install httpx"
            )

        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    async def process_audio_url(
        self,
        audio_url: str,
        user_id: str | None = None,
        user_name: str | None = None,
    ) -> dict[str, Any]:
        """Process audio from a URL and return a formatted result.

        Parameters
        ----------
        audio_url:
            URL of the audio to download and process.
        user_id:
            Discord user ID (for mentions).
        user_name:
            Display name (fallback if user_id not available).

        Returns
        -------
        dict
            Message payload with ``"content"`` and ``"embed"`` fields.
        """
        logger.info("Processing Discord audio (user=%s)", user_id or user_name)

        audio_bytes = await self._download_audio(audio_url)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = await self._engine.process_voice_input(tmp_path)

            if self._format_callback and user_id:
                text = self._format_callback(result, user_id)
            else:
                text = self._format_message(result, user_id, user_name)

            return self._build_discord_message(text, result)

        except IntentEngineError as exc:
            logger.exception("Intent Engine error processing Discord audio")
            return {"content": f"Failed to process audio: {exc}"}
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def process_audio_attachment(
        self,
        attachment: Any,
        channel: Any = None,
        user_id: str | None = None,
        user_name: str | None = None,
    ) -> dict[str, Any]:
        """Process a Discord attachment (from ``discord.py``).

        Parameters
        ----------
        attachment:
            A ``discord.Attachment`` object with a ``.url`` attribute.
        channel:
            Optional ``discord.TextChannel`` for context (not used
            directly but available for subclass overrides).
        user_id:
            Discord user ID.
        user_name:
            Display name.

        Returns
        -------
        dict
            Message payload with ``"content"`` and ``"embed"`` fields.
        """
        url = getattr(attachment, "url", str(attachment))
        return await self.process_audio_url(url, user_id, user_name)

    @staticmethod
    def _format_message(
        result: Result,
        user_id: str | None = None,
        user_name: str | None = None,
    ) -> str:
        """Format the default Discord message."""
        if user_id:
            user_ref = f"<@{user_id}>"
        elif user_name:
            user_ref = f"**{user_name}**"
        else:
            user_ref = "A user"

        return (
            f"[Detected **{result.emotion}**] "
            f"{user_ref} said: {result.text}"
        )

    @staticmethod
    def _build_discord_message(
        text: str,
        result: Result | None = None,
    ) -> dict[str, Any]:
        """Build a Discord message payload with an embed."""
        message: dict[str, Any] = {"content": text}

        if result:
            # Emotion -> color mapping for Discord embeds
            emotion_colors = {
                "joyful": 0xFFD700,    # gold
                "sad": 0x4169E1,       # royal blue
                "angry": 0xFF0000,     # red
                "frustrated": 0xFF4500, # orange-red
                "calm": 0x2E8B57,      # sea green
                "empathetic": 0x9370DB, # medium purple
                "sarcastic": 0xDAA520,  # goldenrod
                "fearful": 0x800080,   # purple
                "surprised": 0xFF69B4, # hot pink
                "disgusted": 0x556B2F, # dark olive green
                "sincere": 0x4682B4,   # steel blue
                "uncertain": 0xA9A9A9, # dark gray
                "neutral": 0x808080,   # gray
            }

            message["embed"] = {
                "color": emotion_colors.get(result.emotion, 0x808080),
                "fields": [
                    {
                        "name": "Emotion",
                        "value": result.emotion,
                        "inline": True,
                    },
                    {
                        "name": "Confidence",
                        "value": f"{result.confidence:.0%}",
                        "inline": True,
                    },
                    {
                        "name": "Suggested Tone",
                        "value": result.suggested_tone,
                        "inline": True,
                    },
                ],
            }

        return message
