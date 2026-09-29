# Intent Engine

**AI that understands not just what you said, but how you meant it.**

![Status](https://img.shields.io/badge/status-beta-yellow)
![Version](https://img.shields.io/badge/version-0.8.0-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

---

## The Problem

Voice AI systems today are **tone-deaf**:

```
Customer: "This is the THIRD TIME I've called about this!"
         [frustrated, angry tone]

Traditional AI hears: "this is the third time i've called about this"

AI Response: "Thank you for calling! How can I help you today?"

Customer: [hangs up, writes angry review]
```

**Current voice AI loses significant emotional context** because speech-to-text strips away prosody (tone, emphasis, pitch, rhythm).

This causes:
- ❌ Misunderstood sarcasm
- ❌ Missed urgency signals  
- ❌ Inappropriate cheerfulness to frustrated customers
- ❌ Constitutional AI that can't verify genuine intent
- ❌ Assistive tech that sounds robotic

---

## The Solution

**Intent Engine** is a prosody-aware AI system that preserves and interprets emotional intent throughout the entire conversation pipeline:

```
Customer: "This is the THIRD TIME I've called!"
         [frustrated tone detected]

Intent Engine hears: 
  <utterance emotion="frustrated" confidence="0.91">
    This is the <emphasis level="strong">THIRD</emphasis> TIME 
    I've called about this!
  </utterance>

AI understands: High frustration, repeated issue, escalation needed

AI Response: "I can hear this has been really frustrating. 
              Let me escalate you to a senior specialist immediately."
              [empathetic tone applied]

Customer: [finally feels heard]
```

> The exchange above illustrates the goal; it is not captured output. What the pipeline reports depends on the recording: pitch, volume and rate are measured against the speaker's own baseline, and an `emotion` is only written when Prosody Protocol's classifier is confident. The built-in classifier can label `calm`, `sad`, `angry`, `joyful` and `fearful` (not `frustrated` or `sarcastic`), and it reports no emotion for a single sentence (a prosody profile can still supply one). See [Emotion detection and abstention](#emotion-detection-and-abstention).

---

## How It Works

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: PROSODY-AWARE STT                             │
│  Speech → Text + Emotional Context                      │
│                                                          │
│  Input:  Audio waveform                                 │
│  Output: Text + IML markup (pauses, emphasis, pitch,    │
│          volume, rate; emotion only when reliable)      │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 2: INTENT INTERPRETATION                         │
│  Text + Prosody → Emotional Intent                      │
│                                                          │
│  LLM prompted to read IML (no fine-tuning yet)          │
│  Weighs cues for: sarcasm, urgency, sincerity, etc.     │
│  Constitutional filter: prosody rules gate actions      │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 3: PROSODY-AWARE TTS                             │
│  Response Text + Emotion → Natural Speech               │
│                                                          │
│  Maps the response emotion to voice settings (rate,     │
│  volume, style; how much depends on the TTS provider)   │
└─────────────────────────────────────────────────────────┘
```

### Powered By

- **[Prosody Protocol](https://github.com/kase1111-hash/Prosody-Protocol)** - IML specification, SDK, and training datasets
- **Mavis** - Generates prosody training data (via Prosody Protocol's `MavisBridge`)
- **Constitutional AI Principles** - Prosody-based rules that gate sensitive actions on how a request was spoken

---

## Features

### 🎯 Emotional Context Preservation

Captures pitch, volume, speaking rate, pauses and emphasis as IML markup (measured against each speaker's own baseline) and prompts the LLM to weigh them:
- **Sarcasm vs. Sincerity** - "Oh great" means different things
- **Urgency Levels** - Fast speech + high pitch = needs immediate help
- **Confidence vs. Uncertainty** - Rising intonation = actually asking, not stating
- **Frustration Markers** - Volume spikes + pitch variation = escalation needed
- **Joy/Enthusiasm** - Positive prosody reinforcement

These are tendencies the LLM is told to weigh as evidence, not detectors. The built-in emotion classifier labels only `calm`, `sad`, `angry`, `joyful` and `fearful`, and cannot hear sarcasm or frustration.

### 🛡️ Constitutional AI Integration

Gates sensitive actions on how they were asked for, before executing them:

```python
# Illustration of the idea (pseudo-code, not the library API).
# The working flow is under "With Constitutional Governance" below.
user_speech = "Yeah just delete everything, that'll help"
prosody_analysis = {
    "emotion": "sarcastic",
    "confidence": 0.89,
    "pitch_contour": "fall-rise",
    "volume": "+6dB"  # Raised voice
}

# Constitutional filter
if prosody_analysis["emotion"] in ["sarcastic", "frustrated"]:
    # User is venting, NOT commanding
    response = "I can tell you're frustrated. What's actually going wrong?"
else:
    # Proceed with confirmation for genuine request
    response = "This will delete all files. Please confirm."
```

The built-in classifier cannot label `sarcastic` or `frustrated`, so the working example below relies on labels it can emit (`calm`, `angry`) and shows what the filter really returns.

### ♿ Accessibility Support

- **Atypical Prosody Profiles** - Describe an individual's prosodic patterns (autism, stroke recovery, etc.) in a JSON profile; see [Prosody profiles](#prosody-profiles)
- **Custom Emotion Mappings** - "When I speak monotone, I'm excited, not bored"
- **Augmentative Communication** - Turn typed text into speech with a chosen emotion (`type_to_speech`)

### 🔌 Platform Integration

Works with:
- **Anthropic Claude** - Prosody understanding via IML-aware system prompts
- **OpenAI GPT** - Via IML-aware system prompts
- **Local Models** - GGUF models through llama.cpp, or any OpenAI-compatible server (Ollama, vLLM), with prosody-aware prompts
- **Existing Voice Platforms** - Twilio, Slack, Discord (example code in `examples/integrations/`, not part of the installed package)

### 🏠 Deployment Flexibility

- **Hybrid** - Cloud STT (and cloud TTS, if you choose one), local LLM
- **Fully Local** - Complete sovereignty (your infrastructure); `engine.is_fully_local` reports whether the chosen providers really are local

---

## Quick Start

### Installation

[Prosody Protocol](https://github.com/kase1111-hash/Prosody-Protocol) (`prosody-protocol`) provides IML parsing, validation, prosody analysis, and emotion classification. It is not published to PyPI yet, so install it from GitHub first, then install Intent Engine from a checkout (Python 3.10 or newer):

```bash
pip install "prosody-protocol[audio] @ git+https://github.com/kase1111-hash/Prosody-Protocol.git"

git clone https://github.com/kase1111-hash/Intent-Engine.git
cd Intent-Engine
pip install -e .            # add provider extras as needed, e.g. ".[claude,whisper,elevenlabs]"
```

Intent Engine requires `prosody-protocol[audio]>=0.1.0a3` and is tested against commit `4d4f0bb930b33f5d66015f8565a87e16c02e5fe2`, the one pinned in `.github/workflows/ci.yml`. Append `@4d4f0bb930b33f5d66015f8565a87e16c02e5fe2` to the GitHub URL above to install exactly that.

Each provider is an optional extra with its own requirements:

| Provider name | Extra | What it needs |
|---|---|---|
| STT `whisper-prosody` | `whisper` | `openai-whisper` (pulls in PyTorch); the `ffmpeg` binary on `PATH` |
| STT `deepgram` | `deepgram` | `deepgram-sdk` 5.x to 7.x; `DEEPGRAM_API_KEY` |
| STT `assemblyai` | `assemblyai` | `assemblyai`; `ASSEMBLYAI_API_KEY` |
| LLM `claude` | `claude` | `anthropic`; `ANTHROPIC_API_KEY` |
| LLM `openai` | `openai` | `openai`; `OPENAI_API_KEY` |
| LLM `local` | `local-llm` (llama.cpp with a GGUF file) or `openai` (an OpenAI-compatible server such as Ollama or vLLM) | no key |
| TTS `elevenlabs` | `elevenlabs` | `elevenlabs`; `ELEVENLABS_API_KEY` |
| TTS `coqui` | `coqui` | the maintained `coqui-tts` fork; it and the original `TTS` are both imported as `TTS`, so install only one, and note that `TTS` 0.22 does not install on Python 3.12 or newer |
| TTS `espeak` | `espeak` | `pyttsx3>=2.99` and the eSpeak NG system library (for example `sudo apt install espeak-ng`) |

The `examples` extra installs what the example integrations need, `dev` the test and lint tools, and `all` everything. The engine builds all three providers when it is created, so a missing API key raises `ValueError` naming the variable (or pass `stt_kwargs`, `llm_kwargs` or `tts_kwargs={"api_key": ...}`). The defaults are `whisper-prosody`, `claude` and `elevenlabs`: a plain `IntentEngine()` needs `ANTHROPIC_API_KEY` and `ELEVENLABS_API_KEY`. Pick `coqui` or `espeak` for TTS if you have no ElevenLabs key.

### Basic Usage

`process_voice_input`, `generate_response`, `synthesize_speech` and `type_to_speech` are coroutines. In a plain script, call their `*_sync` twins:

```python
from intent_engine import IntentEngine

# Initialize with your preferred configuration
engine = IntentEngine(
    stt_provider="whisper-prosody",  # or "deepgram", "assemblyai"
    llm_provider="claude",           # or "openai", "local" (needs llm_kwargs)
    tts_provider="elevenlabs"        # or "coqui", "espeak"
)

# Process a voice conversation turn
audio_file = "customer_complaint.wav"

# Analyze with prosody
result = engine.process_voice_input_sync(audio_file)

print(f"Transcription: {result.text}")
print(f"Detected emotion: {result.emotion} ({result.confidence:.2%})")  # neutral (0.00%): none reported
print(f"IML markup: {result.iml}")
print(f"User's tone: {result.suggested_tone}")

# Generate a response. `tone` is a hint about how the USER sounds,
# not the tone of the reply
response = engine.generate_response_sync(
    result.iml,
    context="customer_support",
    tone=result.suggested_tone
)

# Synthesize the reply in the emotion the LLM chose for it
audio_response = engine.synthesize_speech_sync(
    response.text,
    emotion=response.emotion
)

# "mp3" for ElevenLabs by default, "wav" for eSpeak and Coqui
audio_response.save(f"response.{audio_response.format}")
```

In async code (a web server, a bot), `await` the coroutines instead:

```python
import asyncio

from intent_engine import IntentEngine


async def main() -> None:
    engine = IntentEngine(stt_provider="whisper-prosody", llm_provider="claude", tts_provider="elevenlabs")

    result = await engine.process_voice_input("customer_complaint.wav")
    response = await engine.generate_response(
        result.iml, context="customer_support", tone=result.suggested_tone
    )
    audio_response = await engine.synthesize_speech(response.text, emotion=response.emotion)
    audio_response.save(f"response.{audio_response.format}")


asyncio.run(main())
```

- The `*_sync` wrappers share one background event loop per engine and raise `RuntimeError` when called from a running event loop (async code, Jupyter): `await` the coroutine there. `engine.close()` stops the background loop early; it also stops when the engine is garbage collected.
- `process_voice_input` takes the path of a local audio file (`audio_path`), not a URL. Results are cached in memory (LRU, keyed by the audio's content and the active prosody profile); pass `use_cache=False` to skip the cache, `cache_size=0` to the engine to disable it, and call `engine.clear_cache()` to drop what it holds (transcripts and emotion).
- If prosody analysis fails (audio that is unreadable, too short or sampled too low), the turn continues with text-only IML: `prosody_features` is empty and no emotion is reported. A warning is logged and the result is not cached. Prosody analysis reads WAV, AIFF, FLAC and MP3 directly; OGG/Opus, WebM, M4A and other formats need `ffmpeg` on `PATH`.
- Failures surface as `STTError`, `LLMError` and `TTSError` (all `IntentEngineError`s).

### Emotion detection and abstention

`Result.emotion` and `Result.confidence` are read from the IML that the LLM also sees, so they always agree with it. Prosody Protocol measures each utterance against the speaker's own baseline and leaves `emotion` out of the IML when its classifier is not confident (below 0.5). Without calibration speech, which the engine does not accept yet, it can only judge a recording that holds at least three utterances, most of them at the speaker's usual level. So:

- **A single utterance gets no emotion from the classifier** (a [prosody profile](#prosody-profiles) mapping can still set one). `Result.emotion == "neutral"` with `Result.confidence == 0.0` means *no emotion was reported*, not "measured neutral". `suggested_tone` is then `"neutral"` too.
- The built-in classifier labels only `calm`, `sad`, `angry`, `joyful` and `fearful`. `calm` is its mild version of the low-arousal pattern (a little lower, quieter, slower and flatter than the speaker's own baseline): a heuristic, not a measure of composure. It has not been benchmarked on real speech. Labels such as `frustrated`, `sarcastic`, `sincere` or `uncertain` reach `Result` only through a [prosody profile](#prosody-profiles) or another source of your own.
- `Result.suggested_tone` is `emotion` when `confidence` is at least 0.5, else `"neutral"`. It describes **the user**. It is a hint for `generate_response(tone=...)`, which the LLM may weigh; the tone to *speak the reply in* is `Response.emotion`, chosen by the LLM (an angry caller may need a calm reply).
- `Response.intent` is the intent label the LLM parsed. `Result.intent` stays `None`: `process_voice_input` runs before the LLM.

### With Constitutional Governance

The constitutional filter decides whether a sensitive action may go ahead, given how the request was spoken. Rules are YAML, and the schema is strict: an unknown key, a misspelled section or a wrong-shaped value raises `ValueError` when the file is loaded, so a condition can never be dropped silently.

```yaml
# constitutional_rules.yaml
rules:
  destructive_file_operations:
    triggers:                     # whole words in the intent label, ignoring case and _ or -
      - "delete"                  # also matches delete_all_files, delete_files, ...
      - "remove all"
      - "remove everything"
      - "wipe"
      - "erase"
    required_prosody:             # every listed condition must hold to pass without verification
      emotion: [calm]
      pitch_variance: low         # under 4 semitones of pitch movement within words
      speaking_rate: [3.0, 6.0]   # syllables per second (conversational pace)
    forbidden_prosody:            # blocks the action outright
      emotion: [angry, frustrated, sarcastic]  # only angry can come from the classifier
    verification:
      method: explicit_confirmation
      retries: 2                  # parsed but not enforced: informational only
```

```python
from intent_engine import IntentEngine


def execute(intent: str) -> str:
    return f"running {intent}"                        # your application performs the action


engine = IntentEngine(constitutional_rules="constitutional_rules.yaml")


def handle_turn(audio_path: str) -> str:
    result = engine.process_voice_input_sync(audio_path)           # 1. transcribe + prosody
    response = engine.generate_response_sync(                      # 2. the LLM names the intent
        result.iml, tone=result.suggested_tone
    )
    decision = engine.evaluate_result(response.intent, result)     # 3. gate the action

    if decision.allow:
        return execute(response.intent)
    if decision.requires_verification:                             # ask for confirmation (see below)
        return f"Please confirm ({decision.verification_method}) before I continue."
    return f"I can't do that: {decision.denial_reason}"            # denied outright
```

What `decision` looks like for someone saying "delete all files" (`Decision` fields other than the ones shown are `None` or `False`):

| The request was spoken as | `decision` |
|---|---|
| a single sentence, no profile: no emotion is reported | `allow=False`, `requires_verification=True`, `verification_method="explicit_confirmation"`, `denial_reason="Rule 'destructive_file_operations': Required emotion not met (emotion unknown)"` |
| the last of several sentences, in a voice the classifier reads as `angry` | `allow=False`, `requires_verification=False`, `denial_reason="Rule 'destructive_file_operations': Forbidden emotion detected"` |
| `calm` reported with confidence of at least 0.5 (by the classifier on a longer recording, a profile, or a source of your own), at 3 to 6 syllables per second, with under 4 semitones of pitch movement | `allow=True` |

How the filter decides:

- **Unknown emotion fails closed.** A missing emotion, or one whose confidence is below 0.5 (`("neutral", 0.0)` from a single sentence, for example), fails a required `emotion` list and never counts as calm. A required `pitch_variance` or `speaking_rate` that could not be measured (no prosody features) fails too. With the built-in classifier, an action guarded by a required emotion therefore asks for verification whenever no emotion was reported.
- **A forbidden list only blocks a known emotion.** Pair it with a required `emotion` list, as above, so that an unknown emotion still fails.
- **The most restrictive decision wins** when several rules match: deny, then `two_factor`, then `explicit_confirmation`, then allow, whatever the order of the rules.
- **An intent that matches no rule is allowed, and so is every intent when the engine has no `constitutional_rules`.** The intent label is written by the LLM (a short snake_case label such as `delete_all_files`) and may differ between runs. List every phrasing you care about in `triggers` (whole words: `payment` does not match `payments`), or, for actions that must never slip through, pass the label of the action your code is about to run: `engine.evaluate_result("delete_all_files", result)`.
- **The filter returns a decision; it does not ask for the confirmation itself.** `verification_method` tells your application what to ask for, and `retries` is not counted. Confirm in a channel you control (a button, a typed phrase, a second factor): a spoken "yes" is a single utterance, which reports no emotion, so it would fail the same emotion rule again.
- The detected emotion label never appears in `denial_reason` or in logs above DEBUG, because emotional data is sensitive.

---

## Use Cases

These are scenarios the design targets. Intent Engine supplies the per-turn signals (transcript, IML, an emotion when one can be reported, a constitutional decision); the applications around them (screening, monitoring, tutoring) are not part of this package and none has been evaluated. The dialogues are illustrative.

### 1. Customer Support

**Before Intent Engine:**
```
Customer: "I've been on hold for 20 minutes!" [angry]
AI: "Thank you for your patience! How may I assist you?"
Customer: [angrier]
```

**With Intent Engine (illustrative):**
```
Customer: "I've been on hold for 20 minutes!" [angry detected]
AI: "I'm really sorry about the wait - that's unacceptable. 
     Let me get you help immediately." [empathetic tone]
Customer: [feels heard, stays on line]
```

**Potential impact:** Reduced call escalations and improved customer satisfaction by responding with emotional awareness (not measured).

### 2. Healthcare

Intent Engine contains no clinical models: it does not screen for depression or any other condition, does not track a person over days, and must not be used to diagnose. Prosodic cues are probabilistic evidence and differ between people. A cautious use is to flag a check-in for a person to review:

```python
result = engine.process_voice_input_sync("checkin.wav")

if result.confidence >= 0.5 and result.emotion in ("sad", "fearful"):
    flag_for_human_review(result.text)  # your function; a person decides what it means
```

### 3. Education

Ideas that the prosody measurements and profiles could support; none is implemented.

**Language Learning:**
- Detect non-native prosody patterns
- Provide feedback on intonation
- Help learners sound more natural

**Autism Support:**
- Teach prosody recognition interactively
- Provide real-time feedback on emotional expression
- Build custom prosody vocabularies

### 4. Constitutional Agents

**Intent Verification:**
```python
# The agent is about to act on a spoken request (`agent` is your own object;
# the rules file needs a rule whose triggers match the intent)
result = engine.process_voice_input_sync("request.wav")
response = engine.generate_response_sync(result.iml, tone=result.suggested_tone)

decision = engine.evaluate_result(response.intent, result)
if decision.allow:
    # Delivered in a way the rules accept
    agent.execute(response.intent)
elif decision.requires_verification:
    # Delivery unknown or not as required: confirm in a channel you control
    agent.confirm_first(decision.verification_method)
else:
    # Forbidden delivery: do not proceed
    agent.defer("Let's talk this through first.")
```

### 5. Accessibility

**Non-Verbal Communication:**
```python
# User types text
typed_text = "I'm feeling overwhelmed"

# The emotion label shapes the voice. Use a core label: "stressed" is not one
# and would be spoken neutrally; "fearful" is the closest
speech = engine.type_to_speech_sync(typed_text, emotion="fearful")
speech.save(f"typed.{speech.format}")
```

The text is spoken as typed. How much the emotion changes the voice depends on the TTS provider (eSpeak: rate and volume; ElevenLabs: voice settings; Coqui: speed only, which only some models honour). A speaker's [prosody profile](#prosody-profiles) applies to speech coming in (`process_voice_input`), not to `type_to_speech`.

---

## Architecture Deep Dive

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Intent Engine                        │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  STT Module  │  │  LLM Module  │  │  TTS Module  │ │
│  │              │  │              │  │              │ │
│  │ • Whisper    │  │ • Claude API │  │ • ElevenLabs │ │
│  │ • Deepgram   │  │ • OpenAI     │  │ • Coqui      │ │
│  │ • AssemblyAI │  │ • Local LLM  │  │ • Espeak     │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                 │                 │          │
│         └────────┬────────┴────────┬────────┘          │
│                  │                 │                   │
│         ┌────────▼─────────────────▼────────┐          │
│         │   Prosody Analyzer                │          │
│         │   • Pitch extraction               │          │
│         │   • Energy analysis                │          │
│         │   • Tempo detection                │          │
│         │   • Emotion classification         │          │
│         └────────┬──────────────────────────┘          │
│                  │                                      │
│         ┌────────▼──────────────────────────┐          │
│         │   Constitutional Filter           │          │
│         │   • Allow / verify / deny          │          │
│         │   • Safety checks                  │          │
│         │   • Governance rules               │          │
│         └────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

### STT Module

**Providers:**

| Provider | Runs | Settings | Notes |
|----------|------|----------|-------|
| `whisper-prosody` | Locally (`openai-whisper`) | `model_size` (default `"base"`; `tiny` to `large-v3`), `device`, `language` | Needs the `ffmpeg` binary |
| `deepgram` | Cloud | `model` (default `"nova-2"`), `language` (default `"en"`) | `deepgram-sdk` 5.x to 7.x |
| `assemblyai` | Cloud | `language_code` (default `"en"`) | |

Pass settings as `stt_kwargs={...}`. Every adapter returns words with timings and nothing else: none supplies emotion or sentiment, and prosody comes from Prosody Protocol's `ProsodyAnalyzer` for every provider. Latency, price and accuracy are the vendors' and depend on your account and audio; none has been measured with these adapters.

**Our Approach:**
- Use base STT for transcription (words and timings, converted with `prosody_protocol.alignment`)
- Measure the audio with **ProsodyAnalyzer** and assemble the IML with **IMLAssembler**
- Cache results (LRU, keyed by audio content and prosody profile)
- If the STT returns text without word timings, the whole recording is measured as one span: the IML still carries the words, but no word-level prosody

### LLM Module

**Prosody-Aware Prompting:**

The system prompt is `SYSTEM_PROMPT` in `intent_engine/llm/prompts.py`; its `PROMPT_VERSION` is logged with every call, and a test validates every IML example in it with `IMLValidator`. It teaches the LLM to read the tags the pipeline writes:

| IML | What the prompt says |
|-----|----------------------|
| `<prosody pitch="+15%" volume="+6dB" rate="150%">` | Values are relative to the speaker's own usual voice. A missing attribute means "not marked as unusual" or "not comparable", not monotone or slow |
| `<emphasis level="strong">word</emphasis>` | A word spoken with notable stress |
| `<pause duration="800"/>` | A silence. One at the start of an utterance is the gap since the previous utterance, not hesitation |
| `<utterance emotion="sarcastic" confidence="0.87">` | An estimate with its confidence. **No `emotion` attribute means the emotion was not reliably detected**, not that the speaker is neutral |

It also tells the LLM to treat prosody as probabilistic evidence and not proof (and to ask a short clarifying question when the difference matters), never to read prosody as a sign that someone is lying or telling the truth, and never to base a consequential decision on it alone. The LLM must answer with JSON: `{"intent": "...", "response_text": "...", "suggested_emotion": "..."}`, where `suggested_emotion` is one of the 13 core emotions (any other label becomes `neutral`, with a warning). A reply that is not that JSON raises `LLMError`.

```python
import asyncio

from intent_engine.llm import create_llm_provider

llm = create_llm_provider("claude")  # reads ANTHROPIC_API_KEY (or pass api_key=...)
iml = '<utterance emotion="sarcastic" confidence="0.87">Oh great, another meeting.</utterance>'

interpretation = asyncio.run(llm.interpret(iml, context="customer_support"))
print(interpretation.intent, "|", interpretation.suggested_emotion)
print(interpretation.response_text)
```

Adapters take their model as a setting (`llm_kwargs={"model": ...}`): `claude` defaults to `claude-sonnet-5-5`, `openai` to `gpt-4o`. The `local` provider needs `model_path` (a GGUF file for llama.cpp) or `base_url` (an OpenAI-compatible server such as Ollama at `http://localhost:11434/v1`) and, for a server, `model`.

### TTS Module

**Emotional Speech Synthesis:**

```python
import asyncio

from intent_engine.tts import create_tts_provider

tts = create_tts_provider("elevenlabs")  # reads ELEVENLABS_API_KEY (or pass api_key=...)

# Synthesize with specific emotion
synthesis = asyncio.run(
    tts.synthesize(text="I understand how you feel.", emotion="empathetic")
)
print(synthesis.format, synthesis.sample_rate, len(synthesis.audio_data))
```

The emotion should be one of the 13 core labels (matched case-insensitively); anything else is spoken with the neutral voice (a warning is logged for an unknown label). `EMOTION_VOICE_MAP` in `intent_engine/tts/base.py` maps each label to voice parameters, and each adapter applies what it can:

| Adapter | What the emotion changes |
|---------|--------------------------|
| `espeak` | Speaking rate and volume |
| `elevenlabs` | ElevenLabs voice settings (stability, similarity boost, style) |
| `coqui` | The `speed` argument only. `TTS` 0.22 discards it; `coqui-tts` forwards it, XTTS honours it, and other models (including the default Tacotron2) ignore it, so every emotion sounds alike |

The map's `pitch_shift` is not applied by any built-in adapter. Adapters speak plain text: none of them reads SSML (`supports_ssml` is `False`), so do not pass markup.

### Constitutional Filter

**Rule Definition:**

```yaml
# constitutional_rules.yaml
rules:
  destructive_file_operations:
    triggers:
      - "delete"
      - "remove all"
      - "remove everything"
      - "wipe"
      - "erase"
    required_prosody:
      emotion: [calm]
      pitch_variance: low          # under 4 semitones of pitch movement within words
      speaking_rate: [3.0, 6.0]    # syllables per second (conversational pace)
    forbidden_prosody:
      emotion: [angry, frustrated, sarcastic]
    verification:
      method: explicit_confirmation
      retries: 2                   # informational: not enforced

  financial_transactions:
    triggers:
      - "send money"
      - "transfer"
      - "payment"
    required_prosody:
      emotion: [calm]
    verification:
      method: two_factor
```

The rule keys are `triggers` (required), `required_prosody` and `forbidden_prosody` (conditions `emotion`, `pitch_variance` and `speaking_rate`; only `emotion` can be forbidden) and `verification` (`method`: `explicit_confirmation` or `two_factor`; `retries`). Nothing else is accepted: a key such as `pause_before_amount`, or a voice quality, jitter, shimmer or intensity condition, raises `ValueError` on load instead of being ignored. Emotion labels come from Prosody Protocol's core vocabulary. The classifier only emits `calm`, `sad`, `angry`, `joyful` and `fearful`, so a rule that lists any other label (as `frustrated` and `sarcastic` above) logs a warning and only matches emotions from a [prosody profile](#prosody-profiles) or another source you supply. `neutral` in a `required_prosody` list cannot match a classifier result, because the classifier never reports `neutral` with a confidence of 0.5 or more.

**Runtime Evaluation:**

```python
from intent_engine import ConstitutionalFilter

constitution = ConstitutionalFilter.from_yaml("constitutional_rules.yaml")

# `result` comes from engine.process_voice_input_sync(...)
decision = constitution.evaluate(
    intent="delete_all_files",
    prosody_features=result.prosody_features,
    emotion=result.emotion,
    emotion_confidence=result.confidence,  # ("neutral", 0.0): no emotion reported, counts as unknown
)

if decision.allow:
    print("go ahead")
elif decision.requires_verification:
    print("ask for", decision.verification_method)
else:
    print("denied:", decision.denial_reason)
```

`engine.evaluate_result(intent, result)` does the same with the engine's own filter (`constitutional_rules=`) and is the usual way to gate an action; see [With Constitutional Governance](#with-constitutional-governance) for what the decisions mean. `evaluate` also accepts `context=`, which rules do not use yet, and `min_emotion_confidence=` (default 0.5).

### Prosody Profiles

Some speakers' delivery does not follow the patterns the default classifier expects: a flat voice that is calm rather than bored, fast speech that is excitement rather than anger. A profile is a JSON file, best written with the speaker, that maps prosodic patterns to what they mean for that person:

```json
{
  "profile_version": "1.0.0",
  "user_id": "user_123",
  "description": "Flat delivery is calm; fast flat speech is joy",
  "prosody_mappings": [
    {
      "pattern": {"pitch_contour": "flat", "rate": "fast"},
      "interpretation": {"emotion": "joyful", "confidence_boost": 0.3}
    },
    {
      "pattern": {"pitch_contour": "flat"},
      "interpretation": {"emotion": "calm", "confidence_boost": 0.6}
    }
  ]
}
```

```python
from intent_engine import IntentEngine

engine = IntentEngine(prosody_profile="profile.json")  # validated on load; ProfileError if invalid

profile = engine.load_profile("other_profile.json")    # or build one with engine.create_profile(...)
engine.set_profile(profile)                            # switch at runtime
engine.clear_profile()
```

- Patterns use the vocabulary of `schemas/prosody-profile.schema.json` in the Prosody Protocol repo: `pitch` (`high`, `low`, `normal`), `pitch_contour` (`rise`, `fall`, `rise-fall`, `fall-rise`, `fall-sharp`, `rise-sharp`, `flat`), `volume` (`loud`, `quiet`, `normal`, `spike`), `rate` (`fast`, `slow`, `normal`), `quality` (`modal`, `breathy`, `tense`, `creaky`, `whispery`, `harsh`), and `pause_frequency` and `emphasis_frequency` (`high`, `low`, `normal`). `profile_version` must be a semantic version (`X.Y.Z`). A profile that is invalid, or that uses the older vocabulary (`f0_mean`, `speech_rate`, absolute Hz or dB thresholds), raises `ProfileError`; a failed `set_profile` keeps the previous profile.
- The IML assembler applies the profile to each utterance against the speaker's baseline. Where a mapping matches, its emotion is used and the IML records why, so a single sentence can carry an emotion after all: `<utterance emotion="calm" confidence="0.6" x-profile="pitch_contour=flat">I am fine.</utterance>`. When several mappings match, the more specific pattern (more keys) takes precedence. The profile's `user_id` is neither written into the IML nor logged at INFO.
- A mapping only shows when the classifier's confidence plus its `confidence_boost` reaches 0.5. A single utterance gives the classifier nothing to go on, so only a boost of 0.5 or more applies there; a smaller boost applies on recordings where the classifier already has some evidence.
- A profile decides the emotion for matching speech, so it can satisfy (or dodge) an emotion rule in the constitutional filter. Treat profiles as trusted configuration, not as user input.
- Emotion labels in a profile may be custom (`excitement`). They pass through to the IML (the validator notes them at info level, V15), but the TTS adapters speak an unknown label with the neutral voice.
- Profiles apply to speech coming in; they do not change `type_to_speech`.

---

## Deployment

### Hybrid (Cloud STT, Local LLM)

```python
from intent_engine import HybridEngine

engine = HybridEngine(
    stt_provider="deepgram",  # Cloud (needs DEEPGRAM_API_KEY)
    llm_provider="local",     # Your GPU
    llm_model="models/your-model.gguf",  # a llama.cpp GGUF file
    tts_provider="coqui"      # Local
)
```

**Why Hybrid:**
- Cloud STT (and cloud TTS, if you pick `elevenlabs`) can be higher quality than a local model
- The LLM runs locally: transcripts stay off a cloud LLM, and there is no per-request LLM fee
- A different balance of quality, cost, and control than either extreme

`llm_model` is a `.gguf` path for llama.cpp. To use a model served by Ollama or vLLM instead, pass the server's address and its model name: `HybridEngine(llm_model="llama3", llm_kwargs={"base_url": "http://localhost:11434/v1"})` (a name that is not a `.gguf` file without a `base_url` raises `ValueError`). With a cloud LLM provider, `llm_model` is that provider's model name. `engine.is_llm_local` is `False` for a cloud LLM or a public `base_url`, and a warning is logged at construction. Prosody analysis always runs locally; a cloud STT provider receives your audio and a cloud TTS provider your reply text.

### Fully Local (Sovereignty Mode)

```python
from intent_engine import LocalEngine

engine = LocalEngine(
    stt_model="large-v3",                # Whisper model name
    llm_model="models/your-model.gguf",  # llama.cpp GGUF file, must exist
    tts_provider="coqui",
    tts_model="tts_models/en/ljspeech/tacotron2-DDC",  # Coqui model name
)

# Everything runs on your infrastructure
result = engine.process_voice_input_sync("audio.wav")
print(engine.is_fully_local)  # True
```

Each `*_model` option goes to the setting its provider takes, and impossible configurations fail when the engine is created:
- `stt_model` is the Whisper model size (`tiny` to `large-v3`, not a name such as `whisper-large-v3`) or, for `deepgram`, the Deepgram model. It is a `ValueError` with `assemblyai`.
- `tts_model` is a Coqui model name and needs `tts_provider="coqui"`; eSpeak takes none, and `LocalEngine` uses eSpeak unless you choose `coqui`.
- `llm_model` is a `.gguf` file, or, with `llm_kwargs={"base_url": ...}` pointing at a local Ollama or vLLM server, that server's model name.
- A model file (an absolute or `./` path, or a name ending in `.gguf`) that does not exist raises `FileNotFoundError`; `validate_models=False` skips the check. `prosody_model` is only a label: prosody analysis always uses `prosody_protocol`.
- `is_fully_local` is `False`, with a warning at construction, if you choose a cloud provider or a public `base_url`. Whisper downloads a named model to `~/.cache/whisper` the first time it is used, so the first run needs network access unless the weights are already there.

**Hardware Requirements (estimated, not yet benchmarked):**
- **Minimum:** 16GB RAM, CPU-only
- **Recommended:** 32GB RAM, NVIDIA GPU with 24GB+ VRAM
- **Optimal:** 128GB RAM, high-end NVIDIA GPU(s)

> These are estimates based on the underlying model requirements (Whisper, Llama, etc.) and depend on the models you choose, not measured benchmarks from Intent Engine itself.

---

## Performance Metrics

> **Note:** Nothing in this section has been measured. Intent Engine has not been benchmarked with real audio or real providers: the timing tests in this repository (`tests/test_performance.py`) time mocked providers, and the audio its tests use is synthetic.

### Accuracy

No accuracy figures are claimed. The emotion classifier is Prosody Protocol's rule-based heuristic, which abstains when unsure; upstream checks it against 10 synthetic clips, which catches regressions but says nothing about real speech. There is no sarcasm or urgency classifier, and this repository contains no benchmark dataset. Measured benchmarks on real recordings are on the roadmap.

### Latency (Untested Targets)

**End-to-End (voice input → voice response), design targets only:**

| Configuration | STT | LLM | TTS | Total |
|---------------|-----|-----|-----|-------|
| Hybrid | 300ms | 150ms | 200ms | 650ms |
| Local (GPU) | 400ms | 100ms | 300ms | 800ms |
| Local (CPU) | 800ms | 2000ms | 500ms | 3.3s |

Real latency depends on the providers, models, hardware and length of the audio.

---

## Integration Guides

> **Platform examples:** See `examples/integrations/` for example adapters for Twilio, Slack, Discord, and a FastAPI REST server (importable from a checkout as `examples.integrations.twilio_voice`, `slack_bot`, `discord_bot` and `rest_server`; the `examples` extra installs their dependencies). These are demonstration code, not part of the installed package.

### With Constitutional AI Agents

A sketch for an agent framework: gate every action on the filter. `run`, `deny` and `ask_user_to_confirm` are yours to write.

```python
from intent_engine import IntentEngine

engine = IntentEngine(constitutional_rules="constitutional_rules.yaml")


class ConstitutionalAgent:
    async def execute_command(self, voice_input: str) -> str:
        # Transcribe, then let the LLM name the intent
        result = await engine.process_voice_input(voice_input)
        response = await engine.generate_response(result.iml, tone=result.suggested_tone)

        # Constitutional verification
        decision = engine.evaluate_result(response.intent, result)

        if decision.allow:
            return await self.run(response.intent)
        if decision.requires_verification:
            # Confirm in a channel you control (a button, a typed phrase, a second factor)
            if await self.ask_user_to_confirm(decision.verification_method):
                return await self.run(response.intent)

        return self.deny(reason=decision.denial_reason)
```

---

## Security & Privacy

### Data Handling

**Local/Hybrid Mode:**
- All processing on your infrastructure (fully local) or with cloud STT/TTS only (hybrid); check `is_fully_local` and `is_llm_local`
- The engine does not write your input audio to disk. It keeps results (transcript, IML, emotion, features) in an in-memory cache until `clear_cache()`, or not at all with `cache_size=0`
- Cloud providers keep whatever their own retention policy says: STT vendors receive the audio, ElevenLabs the reply text, a cloud LLM the transcript and IML
- Full sovereignty with local deployment

### Compliance

Intent Engine makes no compliance claim. It has no consent recording, opt-out, retention or deletion features beyond `clear_cache()`, and its IML does not set the `consent` or `processing` attributes. Running it locally keeps audio and transcripts on your infrastructure, which may be a precondition for regimes such as HIPAA or GDPR, but meeting one is up to your deployment.

### Emotional Data Ethics

We treat emotional data as **sensitive PII**. What the code does today:
- Emotion labels and intents are kept out of INFO-and-above logs and out of constitutional `denial_reason`s, with one exception: the TTS adapters log the emotion label they are asked to speak with
- The emotion is optional: nothing downstream requires one, and the pipeline reports none when it cannot tell
- The LLM prompt tells the model never to read prosody as evidence of lying or truthfulness, never to use it to judge or profile the speaker, and never to base a consequential decision on it alone

Goals the code does not enforce yet:
- Explicit user consent capture
- A switch to turn emotional analysis off
- Review and deletion of stored emotional metadata for a user

Project policy:
- ❌ Never sell emotional data
- ❌ Never use for manipulation
- ❌ Never use for deception detection

---

## Roadmap

### Implemented (Beta -- not yet validated with real audio)
- [x] Core STT + prosody analysis pipeline (adapters complete; tests run the real prosody analysis on synthetic audio and the adapters against local fakes, never the live services)
- [x] LLM integration (Claude, OpenAI, local) with prosody-aware prompts
- [x] TTS with emotion-to-voice parameter mapping
- [x] Constitutional filter framework (strict YAML rules; fail-closed allow / verify / deny decisions)
- [x] Hybrid and local deployment engines
- [x] Accessibility profile support (via Prosody Protocol)

### Next Priority: Core Validation
- [ ] End-to-end pipeline validation with real audio files
- [ ] Measured accuracy benchmarks on real recordings (emotion detection)
- [ ] Constitutional filter demo on real recordings

### Future
- [ ] Multi-language support
- [ ] Real-time streaming mode
- [ ] LLM fine-tuning pipeline for prosody understanding
- [ ] Managed cloud service

---

## Contributing

We welcome contributions in:

### Code
- STT/TTS provider integrations
- LLM adapter implementations
- Performance optimizations
- Bug fixes

### Research
- Prosody detection algorithms
- Emotion classification improvements
- Cross-cultural prosody patterns
- Accessibility applications

### Documentation
- Integration guides
- Use case examples
- Deployment tutorials
- Translations

See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.



---

## License

Apache License 2.0. Commercial-friendly, patent-grant included, attribution required.

See [LICENSE](./LICENSE) for full terms.

---

## Citation

If you use Intent Engine in research:

```bibtex
@software{intent_engine_2026,
  title={Intent Engine: Prosody-Aware AI for Emotional Intelligence},
  author={Kase Branham},
  year={2026},
  url={https://github.com/kase1111-hash/Intent-Engine},
  version={0.8.0}
}
```

---

## Prosody Protocol Dependency

Intent Engine is built on the **[Prosody Protocol](https://github.com/kase1111-hash/Prosody-Protocol)** SDK, which provides the canonical implementation of:

- **IML (Intent Markup Language)** - XML-based markup for prosodic information (`<utterance>`, `<prosody>`, `<pause>`, `<emphasis>`, `<segment>`)
- **IML Parser & Validator** - Parse and validate IML documents against the spec (rules V1-V33)
- **Prosody Analyzer** - Extract F0, intensity, jitter, shimmer, HNR from audio using Praat
- **Emotion Classifier** - Rule-based emotion classification from prosodic features, measured against the speaker's own baseline (labels `neutral`, `calm`, `sad`, `angry`, `joyful`, `fearful`; it leaves the emotion out when unsure)
- **IML Assembler** - Build IML documents from STT word alignments and prosody features (the documents carry version `0.1.0`)
- **IML-to-SSML Converter** - Convert IML to SSML for TTS engines
- **Accessibility Profiles** - Load and apply atypical prosody profiles for inclusive design
- **Dataset Tools** - Load, validate, and benchmark prosody-emotion training datasets
- **Mavis Bridge** - Convert data from the Mavis vocal typing game into training datasets

This release of Intent Engine requires `prosody-protocol[audio]>=0.1.0a3` (see [Installation](#installation)). Intent Engine does **not** reimplement any of these components. Instead, it provides the orchestration layer (STT/LLM/TTS provider adapters, constitutional filter, deployment engines) that wires Prosody Protocol's tools into a complete voice AI pipeline.

```python
# Intent Engine uses Prosody Protocol types throughout
from prosody_protocol import IMLParser, IMLValidator, IMLAssembler
from prosody_protocol import ProsodyAnalyzer, SpanFeatures, WordAlignment
from prosody_protocol import IMLDocument, Utterance, Prosody, Emphasis, Pause
from prosody_protocol import ProfileLoader, ProfileApplier
from prosody_protocol import IMLToSSML
```

---

## Acknowledgments

Built on:
- **[Prosody Protocol](https://github.com/kase1111-hash/Prosody-Protocol)** (IML specification and SDK)
- **Mavis** (training data generation, via Prosody Protocol's `MavisBridge`)
- **Constitutional AI** framework (Anthropic)
- Research from computational paralinguistics community

Special thanks to accessibility advocates who guided inclusive design.

---

**The future of AI isn't just hearing your words.**  
**It's understanding your heart.**

🎯 **Listen. Interpret. Respond.**
