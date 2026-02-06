"""Platform integrations -- Twilio, Slack, Discord, REST API.

Provides ready-made adapters for common voice platforms::

    from intent_engine.integrations.twilio import TwilioVoiceHandler
    from intent_engine.integrations.slack import SlackBotHelper
    from intent_engine.integrations.discord import DiscordBotHelper
    from intent_engine.integrations.server import create_app
"""

from intent_engine.integrations.discord import DiscordBotHelper
from intent_engine.integrations.server import create_app
from intent_engine.integrations.slack import SlackBotHelper
from intent_engine.integrations.twilio import TwilioVoiceHandler

__all__ = [
    "TwilioVoiceHandler",
    "SlackBotHelper",
    "DiscordBotHelper",
    "create_app",
]
