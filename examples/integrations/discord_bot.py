"""Discord bot helper for voice channel processing.

Provides ``DiscordBotHelper`` which bridges Discord voice events
with the Intent Engine pipeline.  Processes audio from voice
channels or uploaded files, detects emotion and intent, and posts
results back to a text channel.

Usage (from the repository root, with ``discord.py`` installed)::

    from intent_engine import IntentEngine
    from examples.integrations.discord_bot import DiscordBotHelper

    engine = IntentEngine()
    helper = DiscordBotHelper(engine)

    @bot.event
    async def on_message(message):
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("audio/"):
                payload = await helper.process_audio_attachment(
                    attachment, message.channel, user_id=str(message.author.id)
                )
                await message.channel.send(**payload)

The payload's keys are ``discord.py`` ``send()`` arguments: ``content``, and
when an emotion was reported an ``embed`` (a ``discord.Embed``) and
``allowed_mentions`` (no mention in the text ever notifies anyone).

Emotion is sensitive personal data: this posts a named user's transcript
and inferred emotion into a shared channel.  Tell the people in the channel,
and only run it where they have agreed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from intent_engine.models.result import Result

from ._common import (
    DEFAULT_MAX_DOWNLOAD_BYTES,
    download_media,
    emotion_reported,
    temp_audio_file,
)

logger = logging.getLogger(__name__)

DISCORD_HOSTS = ("cdn.discordapp.com", "media.discordapp.net")
"""Hosts (and their subdomains) attachments are downloaded from by default."""

_MAX_CONTENT_CHARS = 2000
"""Discord rejects a message whose content is longer than this."""


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
        audio.  Defaults to a size-capped ``httpx`` download restricted to
        Discord's CDN hosts; a custom function is responsible for its own
        restrictions.
    max_download_bytes:
        Largest attachment the default downloader accepts.
    allowed_hosts:
        Hosts (and their subdomains) the default downloader may fetch from.
    """

    def __init__(
        self,
        engine: Any,
        format_callback: Callable[[Result, str], str] | None = None,
        download_func: Callable[..., Any] | None = None,
        *,
        max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
        allowed_hosts: Iterable[str] = DISCORD_HOSTS,
    ) -> None:
        self._engine = engine
        self._format_callback = format_callback
        self._download_func = download_func
        self._max_download_bytes = max_download_bytes
        self._allowed_hosts = tuple(allowed_hosts)

    async def _download_audio(self, url: str) -> bytes:
        """Download audio bytes from a URL."""
        if self._download_func:
            result: bytes = await self._download_func(url)
            return result

        return await download_media(
            url,
            allowed_hosts=self._allowed_hosts,
            max_bytes=self._max_download_bytes,
        )

    async def process_audio_url(
        self,
        audio_url: str,
        user_id: str | None = None,
        user_name: str | None = None,
    ) -> dict[str, Any]:
        """Process audio from a URL and return a formatted result.

        Never raises for a bad or undecodable file or a pipeline failure:
        the returned message then says so generically, and the details are
        logged, not posted to the channel.

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
            Keyword arguments for ``channel.send()``: ``"content"`` and,
            when an emotion was reported, ``"embed"`` and
            ``"allowed_mentions"``.
        """
        logger.debug("Processing Discord audio (user=%s)", user_id or user_name)

        try:
            audio_bytes = await self._download_audio(audio_url)

            async with temp_audio_file(audio_bytes) as tmp_path:
                result = await self._engine.process_voice_input(tmp_path)

            if self._format_callback and user_id:
                text = self._format_callback(result, user_id)
            else:
                text = self._format_message(result, user_id, user_name)

            return self._build_discord_message(text, result)

        except Exception:
            # One bad clip must not crash the bot's event handler, and the
            # channel must not see provider or filesystem details.
            logger.exception("Error processing Discord audio")
            return {"content": "Failed to process audio."}

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
            Keyword arguments for ``channel.send()``, as
            :meth:`process_audio_url` returns.
        """
        url = getattr(attachment, "url", str(attachment))
        return await self.process_audio_url(url, user_id, user_name)

    @staticmethod
    def _format_message(
        result: Result,
        user_id: str | None = None,
        user_name: str | None = None,
    ) -> str:
        """Format the default Discord message.

        The emotion is only mentioned when the engine reported one.
        """
        if user_id:
            user_ref = f"<@{user_id}>"
        elif user_name:
            user_ref = f"**{user_name}**"
        else:
            user_ref = "A user"

        detected = f"[Detected **{result.emotion}**] " if emotion_reported(result) else ""
        return f"{detected}{user_ref} said: {result.text}"

    @staticmethod
    def _build_discord_message(
        text: str,
        result: Result | None = None,
    ) -> dict[str, Any]:
        """Build a Discord message payload, ready for ``channel.send(**payload)``.

        The content is cut to Discord's 2000 character limit.  An embed is
        only added when the engine reported an emotion.  Mentions in the
        text render but never notify anyone (``@everyone`` included).
        """
        try:
            import discord
        except ImportError as exc:
            raise ImportError(
                "discord.py is required to build Discord messages. "
                "Install it with: pip install discord.py"
            ) from exc

        if len(text) > _MAX_CONTENT_CHARS:
            text = text[: _MAX_CONTENT_CHARS - 3] + "..."
        message: dict[str, Any] = {
            "content": text,
            "allowed_mentions": discord.AllowedMentions.none(),
        }

        if result is not None and emotion_reported(result):
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

            embed = discord.Embed(colour=emotion_colors.get(result.emotion, 0x808080))
            embed.add_field(name="Emotion", value=result.emotion, inline=True)
            embed.add_field(name="Confidence", value=f"{result.confidence:.0%}", inline=True)
            message["embed"] = embed

        return message
