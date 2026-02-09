# Integration Examples

Example platform adapters for using Intent Engine with common voice platforms.

These are **example code**, not part of the core `intent_engine` package. They demonstrate how to wire Intent Engine into real-world platforms.

## Files

| File | Platform | Description |
|------|----------|-------------|
| `server.py` | FastAPI | REST API server exposing `/process`, `/generate`, `/synthesize` endpoints |
| `twilio.py` | Twilio | Voice webhook handler that processes calls through the pipeline |
| `slack.py` | Slack | Bot helper for processing audio files shared in channels |
| `discord.py` | Discord | Bot helper for processing audio attachments and voice |

## Usage

These examples require additional dependencies:

```bash
# For the REST server
pip install fastapi uvicorn

# For Twilio integration
pip install httpx  # for downloading recordings

# For Slack/Discord
pip install httpx  # for downloading attachments
```

See each file's module docstring for usage examples.
