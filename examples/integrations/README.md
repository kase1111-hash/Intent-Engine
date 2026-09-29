# Integration Examples

Example platform adapters for using Intent Engine with common voice platforms.

These are **example code**, not part of the core `intent_engine` package (they are not included in the wheel). They demonstrate how to wire Intent Engine into real-world platforms. Run and import them from a checkout of the repository, from the repository root:

```python
from examples.integrations.rest_server import create_app
from examples.integrations.twilio_voice import TwilioVoiceHandler
from examples.integrations.slack_bot import SlackBotHelper
from examples.integrations.discord_bot import DiscordBotHelper
```

To use one in your own project, copy the module together with `_common.py` (shared helpers) and change the `from ._common import ...` lines to match where you put it.

## Files

| File | Platform | Description |
|------|----------|-------------|
| `rest_server.py` | FastAPI | REST API server exposing `/process`, `/generate`, `/synthesize` endpoints |
| `twilio_voice.py` | Twilio | Voice webhook handler that processes recordings through the pipeline and answers with TwiML |
| `slack_bot.py` | Slack | Bot helper for processing audio files shared in channels |
| `discord_bot.py` | Discord | Bot helper for processing audio attachments and voice messages |
| `_common.py` | (all) | Shared helpers: size-capped downloads from allow-listed hosts, audio type detection, emotion-abstention check |

The modules are named `*_voice`, `*_bot` and `rest_server` on purpose: a file called `discord.py`, `twilio.py` or `slack.py` would shadow the real SDK whenever its folder is on `sys.path`.

## Installing dependencies

The examples need packages that `intent-engine` does not depend on. Install what the example you use needs, on top of `pip install -e ".[dev]"` (or your normal install of `intent-engine`):

| Example | Packages |
|---------|----------|
| `rest_server.py` | `fastapi`, `uvicorn`, `python-multipart` (uploads; FastAPI fails without it), `httpx` (tests), and `prosody-protocol[api]` (the request-size middleware comes from its REST server; needs 0.1.0a3 or newer) |
| `twilio_voice.py` | `httpx` (downloads recordings), `twilio` (only for `validate_twilio_signature`) |
| `slack_bot.py` | `httpx` (downloads files), `slack_sdk` (only for `verify_signature`, and to post the returned message) |
| `discord_bot.py` | `httpx` (downloads attachments), `discord.py` (builds the message payload) |

```bash
# REST server
pip install fastapi uvicorn python-multipart httpx "prosody-protocol[api]"

# Twilio / Slack / Discord
pip install httpx twilio
pip install httpx slack_sdk
pip install httpx discord.py
```

The examples call `IntentEngine()`, so the engine's own providers must be installed and configured too. The defaults are Whisper for STT, Claude for the LLM and ElevenLabs for TTS: `pip install "intent-engine[whisper,claude,elevenlabs]"` with `ANTHROPIC_API_KEY` and `ELEVENLABS_API_KEY` set. Pass other providers to `IntentEngine(...)` (or set `INTENT_STT_PROVIDER`, `INTENT_LLM_PROVIDER`, `INTENT_TTS_PROVIDER` for the REST server).

**ffmpeg.** WAV, AIFF, FLAC and MP3 are decoded directly. Ogg/Opus (Discord voice messages), WebM and M4A (common Slack clips) need `ffmpeg` on `PATH`; without it those clips fail with a generic "could not process" reply and the reason in the log.

## REST server

```bash
INTENT_API_KEY=change-me \
    uvicorn --factory examples.integrations.rest_server:create_app_from_env
```

`create_app_from_env()` builds the engine from `INTENT_STT_PROVIDER`, `INTENT_LLM_PROVIDER` and `INTENT_TTS_PROVIDER` (unset means the `IntentEngine` defaults) and reads the API key from `INTENT_API_KEY`. To configure things in code, build the app yourself (`uvicorn your_module:app`):

```python
from examples.integrations.rest_server import create_app

app = create_app(stt_provider="deepgram", llm_provider="claude", api_key="change-me")
```

```bash
curl -H "X-API-Key: change-me" -F audio=@clip.wav http://127.0.0.1:8000/process
```

**This is a demo server. Set an API key before exposing it to anything but your own machine.** Every endpoint spends STT/LLM/TTS quota. Without `INTENT_API_KEY` / `api_key=` anyone who can reach the port can use your provider accounts (a warning is logged). Keep the default bind address (`127.0.0.1`), or put the server behind TLS and a rate-limiting reverse proxy; nothing in the example limits how many requests one client makes.

| Status | When |
|--------|------|
| 401 | Missing or wrong `X-API-Key` (when a key is configured; `/health` is the only open path, so `/docs` needs the header too) |
| 413 | Request body over the limit (`max_upload_bytes`, default 25 MiB, for audio; a few MB of JSON) |
| 415 | The upload is not WAV, AIFF, FLAC, MP3, Ogg, WebM or M4A (detected from the content, not the file name) |
| 422 | Empty upload, audio that cannot be decoded, invalid IML, or a text field that is empty/too long |
| 502 | The STT, LLM or TTS provider failed |
| 500 | Anything else |

Error responses never contain exception text (which can hold temp paths, provider URLs or key fragments); they quote an `error id` that is logged next to the real cause. `/process` returns `emotion: "neutral"` with `confidence: 0.0` when the engine reported no emotion.

## Twilio

```python
handler = TwilioVoiceHandler(engine)

# In your webhook route:
if not TwilioVoiceHandler.validate_twilio_signature(url, form, signature, auth_token):
    ...  # respond 403
twiml = await handler.handle_voice(form["RecordingUrl"], form)
```

- Validate `X-Twilio-Signature` on every request. `handle_voice` trusts the recording URL; the default download only accepts `https://*.twilio.com` and at most 25 MiB.
- The reply is spoken with Twilio's `<Say>` voice. `IntentEngine` returns audio bytes, never a URL, and `<Play>` needs a URL, so to play the engine's emotion-mapped TTS voice pass `audio_publisher=`, an async function that stores an `Audio` somewhere Twilio can fetch and returns its URL (for example a route in your own app that serves the bytes). Without a publisher TTS is not called at all.
- If HTTP authentication for recording media is enabled in your Twilio console, pass a `download_func` that authenticates to `api.twilio.com`.

## Slack

```python
helper = SlackBotHelper(engine, bot_token="xoxb-...")

# In your Events API route:
if not SlackBotHelper.verify_signature(raw_body, request.headers, signing_secret):
    ...  # respond 401
message = await helper.process_audio_file(file_url, channel_id, user_id)
client.chat_postMessage(**message)  # slack_sdk WebClient
```

- The bot token is only ever sent to `https://*.slack.com`; other URLs are refused, and downloads are capped at 25 MiB.
- `handle_file_shared_event` needs the event to carry the file's `mimetype` and `url_private_download`. If the `file_shared` payload you receive holds only the file ID, call `files.info` and pass the enriched event. (Not verified against Slack's current event schema when this example was fixed; check yours.)
- Messages are cut to Slack's 3000 character block limit and `&`, `<`, `>` in transcripts are escaped, so a transcript cannot notify a channel.

## Discord

```python
helper = DiscordBotHelper(engine)

payload = await helper.process_audio_attachment(attachment, message.channel, user_id=str(message.author.id))
await message.channel.send(**payload)
```

`payload` holds `send()` keyword arguments: `content` (cut to 2000 characters), `embed` (a `discord.Embed`) when an emotion was reported, and `allowed_mentions`, which switches every mention notification off so an `@everyone` in a name or transcript cannot ping the server. Attachments are only downloaded from `cdn.discordapp.com` / `media.discordapp.net`.

## Emotion is optional

The engine reports `("neutral", 0.0)` when it has no emotion to report. The chat and voice examples only mention an emotion (message text, Slack context block, Discord embed, Twilio escalation replies) when the confidence is at least 0.5; otherwise they show the transcript alone rather than "neutral, 0% confidence".

## Privacy

The Slack and Discord examples post a named user's transcript and the emotion inferred from their voice into a shared channel. Emotional data is sensitive personal data: tell the people who use the channel, get their agreement, and do not run these where people have not opted in. The Slack and Discord examples log user and channel identifiers at DEBUG level only.

## Tests

```bash
pip install -e ".[dev]"
pytest examples
```

Tests that need an optional package (`fastapi`, `httpx`, `twilio`, `slack_sdk`, `discord.py`, `uvicorn`) skip when it is not installed; everything runs against stub engines and in-process fake HTTP transports, so no provider API key or network access is needed. `pytest examples` is separate from the package's own `pytest` run, which only collects `tests/`.
