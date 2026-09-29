# Intent Engine - Technical Specification

**Version:** 0.8.0 (Beta)
**Author:** Kase Branham
**License:** Apache License 2.0

---

## 1. Overview

Intent Engine is a prosody-aware AI system that preserves and interprets emotional intent throughout a voice conversation pipeline. It bridges the gap between what a user says (words) and how they mean it (tone, emphasis, pitch, rhythm), enabling AI systems to respond with appropriate emotional intelligence.

### 1.1 Problem Statement

Current voice AI systems lose significant emotional context because speech-to-text strips away prosody. This leads to:

- Misunderstood sarcasm
- Missed urgency signals
- Inappropriate responses to frustrated users
- Constitutional AI that cannot verify genuine intent
- Robotic-sounding assistive technology

### 1.2 Goals

- Preserve emotional context across the full speech-to-response pipeline
- Enable AI systems to distinguish sarcasm, urgency, frustration, sincerity, and other emotional states
- Provide constitutional intent verification using prosodic features
- Support accessible and inclusive prosody profiles
- Offer flexible deployment: hybrid and fully local

These are design goals: Sections 3.5 and 9 say what is implemented and what has been measured today.

---

## 2. Architecture

### 2.1 Three-Layer Pipeline

The system processes voice input through three sequential layers:

```
Audio Input
    │
    ▼
┌──────────────────────────────────────┐
│  Layer 1: Prosody-Aware STT          │
│  Audio → Text + IML Markup           │
│  Providers: Whisper, Deepgram,       │
│             AssemblyAI               │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Layer 2: Intent Interpretation      │
│  IML Markup → Emotional Intent       │
│  Providers: Claude, OpenAI, Local    │
│  Includes: Constitutional Filters    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Layer 3: Prosody-Aware TTS          │
│  Response + Emotion → Natural Speech │
│  Providers: ElevenLabs, Coqui,       │
│             eSpeak                   │
└──────────────────────────────────────┘
    │
    ▼
Audio Output
```

### 2.2 Core Components

| Component | Responsibility |
|---|---|
| **STT Module** | Transcribes audio to words with timings (prosody comes from the Prosody Analyzer) |
| **Prosody Analyzer** | Extracts pitch, energy, tempo; classifies emotion (Prosody Protocol; the emotion is left out when it is not confident) |
| **LLM Module** | Interprets intent using prosody-aware prompts |
| **Constitutional Filter** | Gates sensitive actions on prosody rules: allow, ask for verification, or deny |
| **TTS Module** | Synthesizes speech with appropriate emotional tone |
| **IntentEngine (Orchestrator)** | Coordinates the full pipeline end-to-end |

---

## 3. Intent Markup Language (IML)

IML is an XML-based markup language that carries prosodic information through the pipeline. The IML specification, XML Schema (XSD), parser, validator, and data models are defined and maintained by the **[Prosody Protocol](https://github.com/kase1111-hash/Prosody-Protocol)** project. Intent Engine consumes IML via the `prosody_protocol` SDK and does not maintain its own IML implementation.

**Canonical references** (all in the Prosody Protocol repo, version 0.1.0a3):
- IML specification: `spec.md` (IML 0.1.0-alpha, a draft). The IML documents Intent Engine writes carry `version="0.1.0"`
- IML XSD Schema: `schemas/iml-1.0.xsd` (schema version 1.0.0, describing IML 0.1.0-alpha)
- Prosody profile schema: `schemas/prosody-profile.schema.json`
- Parser: `prosody_protocol.IMLParser`
- Validator: `prosody_protocol.IMLValidator` (rules V1-V33: spec violations are errors, SHOULD violations warnings, notes such as V15 are info)
- Data models: `prosody_protocol.models` (`IMLDocument`, `Utterance`, `Prosody`, `Pause`, `Emphasis`, `Segment`)

### 3.1 Elements

| Element | Purpose | Attributes |
|---|---|---|
| `<iml>` | Root wrapper for multi-utterance documents | `version`, `language`, `consent`, `processing` (the pipeline sets `version` and `language` only) |
| `<utterance>` | Wraps a full spoken turn | `emotion`, `confidence`, `speaker_id` |
| `<prosody>` | Marks prosodic features on a span | `pitch`, `pitch_contour`, `volume`, `rate`, `quality` + extended attrs |
| `<pause>` | Explicit timing gap | `duration` (ms, required, positive integer) |
| `<emphasis>` | Marks stressed words | `level` (strong, moderate, reduced) |
| `<segment>` | Clause-level prosodic grouping | `tempo`, `rhythm` (direct child of `<utterance>` only) |

### 3.2 Examples

An illustration of the markup (the built-in classifier does not emit `frustrated`; see Section 3.5):

```xml
<utterance emotion="frustrated" confidence="0.91">
  This is the <emphasis level="strong">THIRD</emphasis> TIME
  I've called about this!
</utterance>
```

What the pipeline itself wrote for a synthetic three-sentence recording (two even sentences, then a louder, higher one):

```xml
<iml version="0.1.0" language="en">
  <utterance>Well that is what we said.</utterance>
  <utterance><pause duration="810"/>Well that is what we said.</utterance>
  <utterance emotion="angry" confidence="0.54"><pause duration="810"/><prosody pitch="+36%" volume="+10dB">Well that is what we said.</prosody></utterance>
</iml>
```

The first two utterances carry no emotion (the classifier was not confident). Pitch and volume are relative to the speaker's own baseline, and a leading `<pause>` is the silence since the previous utterance.

### 3.3 Prosodic Features Captured

**Core attributes (on `<prosody>`):**

| Attribute | Description | Value Format |
|---|---|---|
| `pitch` | Fundamental frequency shift | `+N%`, `-N%`, `+Nst`, `-Nst`, or `NHz` |
| `pitch_contour` | Pitch trajectory pattern | `rise`, `fall`, `rise-fall`, `fall-rise`, `fall-sharp`, `rise-sharp`, `flat` |
| `volume` | Loudness relative to baseline | `+NdB`, `-NdB` |
| `rate` | Speaking rate | `fast`, `slow`, `medium`, or `N%` |
| `quality` | Voice quality, judged relative to the speaker | `modal`, `breathy`, `creaky`, `harsh` (the analyzer never produces `tense` or `whispery`, which the IML spec also allows) |

**Extended attributes (on `<prosody>`, for research use):**

| Attribute | Type | Description |
|---|---|---|
| `f0_mean` | float | Mean fundamental frequency (Hz) |
| `f0_range` | string | Pitch range, e.g., `"120-240"` |
| `f0_contour` | string | Comma-separated Hz values |
| `intensity_mean` | float | Mean intensity (dB) |
| `intensity_range` | float | Dynamic range (dB) |
| `speech_rate` | float | Syllables per second |
| `duration_ms` | int | Span duration in milliseconds |
| `jitter` | float | Cycle-to-cycle frequency perturbation (percent) |
| `shimmer` | float | Cycle-to-cycle amplitude perturbation (percent) |
| `hnr` | float | Harmonics-to-noise ratio (dB) |

**Utterance-level attributes:**

| Attribute | Description | Value Type |
|---|---|---|
| `emotion` | Classified emotional state | String (see Section 3.4) |
| `confidence` | Emotion classification confidence (REQUIRED when emotion is set) | Float 0.0-1.0 |
| `speaker_id` | Speaker identifier | String |

### 3.4 Supported Emotions

**Prosody Protocol core vocabulary (used for validation):**
- `neutral`, `sincere`, `sarcastic`
- `frustrated`, `joyful`, `uncertain`
- `angry`, `sad`, `fearful`
- `surprised`, `disgusted`
- `calm`, `empathetic`

**Custom labels:** any other label is valid IML but is reported by the validator at info level (V15). A prosody profile can produce them (for example `excitement`). Intent Engine defines no extended vocabulary of its own: labels such as `confident`, `deliberate`, `stressed` or `rushed` are not recognised, the TTS adapters speak them with the neutral voice, and the LLM adapters map any label outside the core set to `neutral`.

**Planned (v1.0):** 20+ fine-grained emotions with standardization across both projects.

### 3.5 Emotion Reporting

`Result.emotion` and `Result.confidence` are read from the assembled IML: they are those of the utterance with the highest confidence among those that carry an emotion (the later one on a tie).

- Prosody Protocol measures each utterance against the speaker's own baseline and omits `emotion` and `confidence` below its reporting threshold (0.5). Without calibration speech, which `IntentEngine` does not accept, it can only judge a recording of at least three utterances, most of them at the speaker's usual level. A single utterance therefore gets no emotion from the classifier (a profile mapping can still set one, Section 11.1).
- When the IML carries no emotion, `Result.emotion` is `"neutral"` and `Result.confidence` is `0.0`: *no emotion was reported*, which is not the same as a measured neutral. `Result.suggested_tone` is then `"neutral"`.
- The built-in `RuleBasedEmotionClassifier` labels only `neutral`, `calm`, `sad`, `angry`, `joyful` and `fearful`, and never reports `neutral` at 0.5 or more. `calm` is its mild version of the low-arousal pattern (a little lower, quieter, slower and flatter than the speaker's baseline). It is a heuristic that has been checked only on synthetic speech and cannot hear sarcasm or frustration. Other labels reach `Result` only through a prosody profile (Section 11.1) or another source supplied by the caller.
- `Result.suggested_tone` is `emotion` when `confidence >= 0.5`, else `"neutral"`. It describes the *user*. The tone to speak the reply in is `Response.emotion`, chosen by the LLM.
- If prosody analysis fails (unreadable, too short or too low-sampled audio), the turn continues with text-only IML: `prosody_features` is `[]` and no emotion is reported.

---

## 4. Public API

### 4.1 Main Entry Points

#### IntentEngine (General Purpose)

```python
from intent_engine import IntentEngine

engine = IntentEngine(
    stt_provider="whisper-prosody",    # "whisper-prosody" | "deepgram" | "assemblyai"
    llm_provider="claude",             # "claude" | "openai" | "local" (needs llm_kwargs)
    tts_provider="elevenlabs"          # "elevenlabs" | "coqui" | "espeak"
)
```

**Constructor options:**

| Parameter | Default | Description |
|---|---|---|
| `stt_provider`, `llm_provider`, `tts_provider` | `"whisper-prosody"`, `"claude"`, `"elevenlabs"` | Provider names as above. The engine builds all three when it is created. A provider that needs an API key raises `ValueError` if the key is in neither its environment variable (`DEEPGRAM_API_KEY`, `ASSEMBLYAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`) nor the matching `*_kwargs["api_key"]` |
| `constitutional_rules` | `None` | Path to a YAML rules file (Section 4.4). `None` means no filter: every action is allowed. An empty string is rejected with `ValueError` |
| `prosody_profile` | `None` | Path to a prosody profile JSON (Section 11.1), validated at construction (`ProfileError`) |
| `cache_size` | `128` | Results kept per audio content and active profile (LRU). `0` or less disables caching |
| `stt_kwargs`, `llm_kwargs`, `tts_kwargs` | `None` | Provider-specific keyword arguments, for example `{"model_size": "small"}`, `{"model": "gpt-4o"}` or `{"voice_id": "..."}` |

**Async and sync.** `process_voice_input`, `generate_response`, `synthesize_speech` and `type_to_speech` are coroutines. Each has a `*_sync` twin (`process_voice_input_sync`, `generate_response_sync`, `synthesize_speech_sync`, `type_to_speech_sync`) for plain scripts. The twins share one background event loop per engine and raise `RuntimeError` when called from a running event loop (call the coroutine with `await` there). `close()` stops that loop early; `clear_cache()` drops cached results, which hold transcripts and emotion.

#### HybridEngine (Cloud STT + Local LLM)

```python
from intent_engine import HybridEngine

engine = HybridEngine(
    stt_provider="deepgram",
    llm_provider="local",
    llm_model="models/your-model.gguf",
    tts_provider="coqui"
)
```

`llm_model` is a `.gguf` file for llama.cpp or, with `llm_kwargs={"base_url": "http://localhost:11434/v1"}`, a model name on a local OpenAI-compatible server (Ollama, vLLM); with a cloud LLM provider it is that provider's model name. `is_llm_local` is `False` for a cloud LLM or a public `base_url`, and a warning is logged at construction. The defaults are `deepgram`, `local` and `coqui`; the `local` LLM needs `llm_model` or a `base_url`, or the engine raises `ValueError`.

#### LocalEngine (Full Sovereignty)

```python
from intent_engine import LocalEngine

engine = LocalEngine(
    stt_model="large-v3",
    llm_model="models/your-model.gguf",
    tts_provider="coqui",
    tts_model="tts_models/en/ljspeech/tacotron2-DDC"
)
```

The model options go to the setting each provider takes, and impossible combinations raise `ValueError` at construction: `stt_model` is the Whisper model size (`tiny` to `large-v3`) or the Deepgram model, `tts_model` a Coqui model name (it needs `tts_provider="coqui"`; the default is eSpeak, which takes none), `llm_model` a `.gguf` file or, with `llm_kwargs={"base_url": ...}`, a local server's model name. A model file that does not exist raises `FileNotFoundError` (`validate_models=False` skips the check). `prosody_model` is only a label. `is_fully_local` is `False`, with a warning, when a provider is a cloud service or the LLM's `base_url` is public.

### 4.2 Core Methods

#### `async process_voice_input(audio_path, use_cache=True) -> Result`

Processes an audio file through STT and prosody analysis.

**Parameters:**
- `audio_path` (str): Path to a local audio file (WAV, MP3, etc.); URLs are not supported (`FileNotFoundError`)
- `use_cache` (bool): Use the result cache

**Returns:** `Result` object (a frozen dataclass) with:

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Plain text transcription |
| `emotion` | `str` | Emotion reported in the IML; `"neutral"` with `confidence` 0.0 when none was reported (Section 3.5) |
| `confidence` | `float` | Emotion confidence (0.0-1.0); 0.0 when no emotion was reported |
| `iml` | `str` | Full IML markup of the recording |
| `iml_document` | `IMLDocument` | The same IML as a `prosody_protocol` document |
| `suggested_tone` | `str` | Tone of the *user's* voice worth acting on: `emotion` when `confidence >= 0.5`, else `"neutral"` |
| `prosody_features` | `list[SpanFeatures]` | Per-span prosodic features (`prosody_protocol.SpanFeatures`); `[]` when prosody analysis failed |
| `intent` | `str \| None` | Always `None` here, because the LLM has not run yet; use `Response.intent` |

**Raises:** `FileNotFoundError` (no such file), `ValueError` (not a file), `STTError` (transcription failed), `IntentEngineError` (the IML failed validation), `prosody_protocol.ProsodyProtocolError` (assembling the IML failed, for example STT text with control characters). Audio that cannot be analysed does not raise: the turn continues with text-only IML.

#### `async generate_response(iml, context=None, tone=None) -> Response`

Generates an LLM response using IML-annotated input.

**Parameters:**
- `iml` (str): IML-annotated input text (normally `Result.iml`)
- `context` (str, optional): Conversation context (e.g., `"customer_support"`)
- `tone` (str, optional): The tone of the *user's* voice, normally `Result.suggested_tone`. It is passed to the LLM as a hint and does not set the tone of the reply, which the LLM chooses from what the user needs (an angry caller may need a calm reply)

**Returns:** `Response` object (a frozen dataclass) with:

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Response text |
| `emotion` | `str` | Emotion to apply in TTS (one of the core emotions) |
| `intent` | `str \| None` | The intent label the LLM parsed, for the constitutional filter |

**Raises:** `LLMError` (the call failed, or the reply was not the required JSON).

#### `async synthesize_speech(text, emotion="neutral") -> Audio`

Synthesizes speech with the specified emotional tone.

**Parameters:**
- `text` (str): Text to synthesize (plain text; the built-in adapters do not read SSML)
- `emotion` (str): Emotional tone to apply; a label outside the core vocabulary is spoken neutrally, with a warning

**Returns:** `Audio` object with `data` (bytes), `format` (`"wav"`, `"mp3"`, ...), `sample_rate`, `duration` (seconds, or `None`), `url` (`None`: none of the built-in adapters sets it) and `save(path)`. Save with the extension `format` names.

**Raises:** `TTSError`.

#### `async type_to_speech(text, emotion="neutral") -> Audio`

Speaks typed text as typed, with the given emotion (augmentative communication, Section 11.2). A TTS provider that reads SSML (`supports_ssml = True`; none of the built-in adapters does) is given SSML predicted from the text instead.

#### Other methods

| Method | Description |
|---|---|
| `evaluate_result(intent, result, context=None) -> Decision` | Gate an action on a `Result` (Section 4.3) |
| `evaluate_intent(intent, prosody_features, emotion=None, context=None, *, emotion_confidence=None, min_emotion_confidence=0.5) -> Decision` | The same with the values passed separately |
| `load_profile(path)`, `set_profile(profile)`, `clear_profile()`, `create_profile(user_id, mappings, description=None, profile_version="1.0.0")`, `validate_profile(profile)` | Prosody profile management (Section 11.1) |
| `clear_cache()`, `close()` | Drop cached results; stop the `*_sync` background loop |

### 4.3 Constitutional Filter

```python
from intent_engine import ConstitutionalFilter

constitution = ConstitutionalFilter.from_yaml("constitutional_rules.yaml")

decision = constitution.evaluate(
    intent="delete_all_files",
    prosody_features=result.prosody_features,
    emotion=result.emotion,
    emotion_confidence=result.confidence,
)
```

`evaluate(intent, prosody_features, emotion=None, context=None, *, emotion_confidence=None, min_emotion_confidence=0.5)`. `emotion_confidence` and `min_emotion_confidence` are keyword-only. `context` is accepted for forward compatibility; rules do not use it yet. `ConstitutionalFilter(rules)` takes an iterable of `ConstitutionalRule` objects (a dict or string raises `TypeError`); `from_yaml` is the usual constructor. In the engine, `IntentEngine(constitutional_rules=path).evaluate_result(intent, result)` passes `result.emotion` and `result.confidence` for you, and `Response.intent` is the intent to pass.

**Decision object** (a frozen dataclass):

| Field | Type | Description |
|---|---|---|
| `allow` | `bool` | Whether the action is permitted |
| `requires_verification` | `bool` | Whether additional confirmation is needed |
| `verification_method` | `str \| None` | Method to use (`"explicit_confirmation"` or `"two_factor"`) |
| `denial_reason` | `str \| None` | Human-readable reason for denial; names the rule and the failed condition, never the detected emotion |

**Evaluation semantics:**

1. Triggers match the intent as whole word sequences, ignoring case and `_`, `-` and other separators: the trigger `delete all` matches `delete_all_files` and `DeleteAllFiles`, but not `delete_files` or `undelete_all`, and an intent that is only part of a trigger (`delete` for the trigger `delete all`) does not match. Inflected forms (`payments` for `payment`) are separate words. An intent that matches no rule is allowed, and so is every intent when the engine has no rules.
2. **Unknown emotion fails closed.** The emotion is unknown when it is missing or blank, or its `emotion_confidence` is below `min_emotion_confidence`. An unknown emotion fails a required `emotion` list, and never matches a forbidden one. A required `pitch_variance` or `speaking_rate` that could not be measured (no features, or no usable values) also fails.
3. A rule whose `forbidden_prosody` matches is a hard deny. Otherwise, if `required_prosody` fails, the decision is a verification request when the rule has a `verification` block, and a hard deny when it has none. Otherwise the rule allows.
4. When several rules match, the most restrictive decision wins: hard deny, then `two_factor`, then `explicit_confirmation`, then allow, whatever the order of the rules.
5. Verification is a request to the caller: the filter neither asks for the confirmation nor counts `retries`. A spoken confirmation is a single utterance and reports no emotion, so confirm in a channel the caller controls.
6. The detected emotion label is not written to logs above DEBUG or into `denial_reason`.

The intent label is written by the LLM (see Section 6.1), so it can vary between runs. Triggers must cover the phrasings that matter; for an action that must never slip through, pass the label of the action the code is about to run instead of the LLM's label.

### 4.4 Constitutional Rules Schema (YAML)

```yaml
rules:
  destructive_file_operations:      # rule name: a unique, non-empty string
    triggers:                       # required: at least one phrase
      - "delete"
    required_prosody:               # optional: every listed condition must hold
      emotion: [calm]               #   labels, compared case-insensitively
      pitch_variance: low           #   low | normal | high
      speaking_rate: [3.0, 6.0]     #   [min, max] in syllables per second
    forbidden_prosody:              # optional: blocks the action; only emotion is supported
      emotion: [angry]
    verification:                   # optional: without it a failed required_prosody is a hard deny
      method: explicit_confirmation #   explicit_confirmation | two_factor
      retries: 2                    #   integer >= 0; informational, not enforced
```

| Key | Meaning |
|---|---|
| `triggers` | Phrases matched against the intent (see Section 4.3). A bare string, an empty list or a phrase without words is rejected |
| `emotion` | Emotion labels. Use the core vocabulary (Section 3.4); a label outside it, or one the built-in classifier never emits (it emits `calm`, `sad`, `angry`, `joyful`, `fearful`), is logged as a warning and only matches emotions supplied from another source such as a prosody profile |
| `pitch_variance` | Pitch movement within words, in semitones (10th to 90th percentile of the F0 contour, averaged over the spans): `low` is under 4, `normal` 4 to under 8, `high` 8 or more. These bounds were checked only on synthetic speech |
| `speaking_rate` | `[min, max]` in **syllables per second** (the unit of `SpanFeatures.speech_rate`; conversational speech is roughly 3-6), not words per minute and not a ratio to a normal pace |
| `verification.method` | `explicit_confirmation` or `two_factor` |
| `verification.retries` | Parsed and validated, otherwise unused: `Decision` has no field for it |

**The schema is strict.** An unknown key at any level, a misspelled section, an empty section, a wrong-shaped or out-of-range value, a duplicate rule name, an empty `rules` mapping, an unknown top-level key, or a `pitch_variance` or `speaking_rate` under `forbidden_prosody` raises `ValueError` (naming the file and the rule) when the rules are loaded, so a safety condition is never silently dropped. Voice quality, jitter, shimmer, intensity and pause conditions are not available: they are unknown keys.

---

## 5. STT Provider Specifications

| Provider | Runs | Extra | Settings (defaults) | Credential |
|---|---|---|---|---|
| `whisper-prosody` | Local (`openai-whisper`) | `whisper` | `model_size` (`"base"`; `tiny` to `large-v3`), `device` (`"cpu"`), `language` (auto-detect) | None; needs the `ffmpeg` binary on `PATH` |
| `deepgram` | Cloud | `deepgram` (`deepgram-sdk` 5.x to 7.x) | `model` (`"nova-2"`), `language` (`"en"`) | `DEEPGRAM_API_KEY` |
| `assemblyai` | Cloud | `assemblyai` | `language_code` (`"en"`) | `ASSEMBLYAI_API_KEY` |

**Contract:** `async transcribe(audio_path) -> TranscriptionResult(text, alignments, language)`, where `alignments` is a list of `prosody_protocol.WordAlignment` (`word`, `start_ms`, `end_ms`, and `speaker` when the provider labels speakers). Every adapter returns words and timings only: none supplies emotion or sentiment. Failures raise `STTError` (a missing file raises `FileNotFoundError`, a missing SDK `ImportError`). Blocking SDK and model calls run in worker threads, off the event loop.

Latency, price and accuracy belong to the vendors and depend on the account and the audio; none has been measured with these adapters.

**Processing approach:** Use base STT for words and timings (converted with `prosody_protocol.alignment`), then measure the audio with the ProsodyAnalyzer and assemble the IML with the IMLAssembler. Results are cached (LRU, keyed by audio content and prosody profile). If the STT returns text without word timings, the whole recording is measured as one span, so the IML carries the words but no word-level prosody.

---

## 6. LLM Integration

### 6.1 Prompting Strategy

The LLM receives a system prompt (`SYSTEM_PROMPT` in `intent_engine/llm/prompts.py`; its `PROMPT_VERSION` is logged with every call) that teaches it to interpret IML annotations. The prompt defines:

- How to read IML markup tags, and that pitch, volume and rate are relative to the speaker's own baseline
- How to interpret prosodic features (pitch contours, emphasis, pauses, emotion tags) as tendencies, for example that a fall-rise contour on agreeable words suggests sarcasm or reluctance
- That a missing `emotion` attribute means the emotion was not reliably detected, not that the speaker is neutral
- Response guidelines: serve what the user needs, ask a short clarifying question when prosody and words disagree and it matters, never treat prosody as evidence of lying or truthfulness, and never base a consequential decision on it alone

**Reply contract:** a JSON object with `intent` (a short snake_case label such as `request_help`), `response_text`, and `suggested_emotion` (one of the 13 core emotions; an adapter maps any other label to `neutral`). Fenced or padded JSON is tolerated; any other reply raises `LLMError`. `JSON_RESPONSE_SCHEMA` documents the contract and is not sent to providers. Every IML example and attribute value in the prompt validates with `IMLValidator`, which `tests/llm/test_prompt_iml_conformance.py` enforces.

**Adapters:**

| Provider | Extra | Defaults |
|---|---|---|
| `claude` | `claude` (`anthropic>=0.40`) | `model=` (set it explicitly; see the adapter default), `max_tokens=8192`; no `temperature` is sent unless you set one |
| `openai` | `openai` (`openai>=1.56`) | `model="gpt-4o"`, `max_tokens=1024`, `temperature=0.3` |
| `local` | `local-llm` (`llama-cpp-python`) for `model_path`, or `openai` for `base_url` | `model="llama3"` (server), `n_ctx=4096`, `max_tokens=1024`, `temperature=0.3`; needs `model_path` (a GGUF file) or `base_url` |

Override them with `llm_kwargs`. The adapters are tested against local fake servers and stand-ins; they have not been run against the live services.

### 6.2 Fine-Tuning

Fine-tuning is future work (Section 14). This repository contains no training code, and no fine-tuned model or dataset. A model tuned elsewhere can be used through the `local` provider like any other, provided it follows the reply contract in Section 6.1.

---

## 7. TTS Emotional Synthesis

The TTS module applies emotional parameters to speech output. A provider implements `async synthesize(text, emotion="neutral") -> SynthesisResult(audio_data, format, sample_rate, duration)`.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | Plain text to speak. The built-in adapters do not read SSML (`supports_ssml` is `False`); eSpeak and Coqui reduce a complete `<speak>` document to its text |
| `emotion` | `str` | One of the 13 core emotions, matched case-insensitively. Anything else is spoken with the neutral voice (a warning is logged for an unknown label) |

`EMOTION_VOICE_MAP` (`intent_engine/tts/base.py`) gives each emotion a `pitch_shift` (a string such as `"+10%"`), a `rate` multiplier (1.0 = normal), a `volume_db` offset and style notes. Each adapter applies what it can:

| Adapter | What the emotion changes |
|---|---|
| `espeak` | Speaking rate (times `rate_wpm`, default 175) and volume (the constructor's `volume` is the level of the loudest emotion, so neutral speech is quieter) |
| `elevenlabs` | Its own per-emotion voice settings (`stability`, `similarity_boost`, `style`, in `ELEVENLABS_EMOTION_SETTINGS`). `output_format` (default `mp3_44100_128`) sets the reported `format` and `sample_rate`; `pcm`, `ulaw` and `alaw` outputs are raw samples without a container header |
| `coqui` | The `speed` argument only. `TTS` 0.22 discards it; `coqui-tts` forwards it, XTTS honours it and other models (including the default Tacotron2) ignore it, so every emotion sounds alike |

No built-in adapter applies `pitch_shift`. `espeak` needs the eSpeak NG system library and `pyttsx3>=2.99` and raises `TTSError` when the engine produces no audio; it has been exercised on Linux only.

---

## 8. Deployment Modes

### 8.1 Hybrid

- STT in the cloud (Deepgram or AssemblyAI, for quality); TTS local (Coqui by default) or cloud (`elevenlabs`)
- LLM runs locally (llama.cpp, or an OpenAI-compatible server on this machine or network): transcripts stay off a cloud LLM and there is no per-request LLM fee
- A different balance of quality, cost, and control than either extreme
- `is_llm_local` reports whether the LLM really is local

### 8.2 Fully Local (Sovereignty Mode)

- All providers run on user infrastructure: `whisper-prosody`, a local LLM, and `coqui` or `espeak`
- With `is_fully_local` true, no audio, transcript or reply text is sent to a cloud service. Whisper fetches a named model on first use into `~/.cache/whisper`, so the first run needs network access unless the weights are already there

**Hardware Requirements** (estimates from the underlying model requirements, mirrored in `HARDWARE_TIERS` in `intent_engine/local_engine.py`; not measured, and dependent on the models you choose):

| Tier | RAM | GPU | Performance |
|---|---|---|---|
| Minimum | 16 GB | CPU-only | Slow |
| Recommended | 32 GB | NVIDIA RTX 4090 | Good |
| Optimal | 128 GB | 2x NVIDIA A100 | Best |

---

## 9. Performance Targets

> **Note:** Nothing in this section has been measured. Intent Engine has not been benchmarked with real audio or real providers: the timing tests in this repository (`tests/test_performance.py`) time mocked providers, and the audio its tests use is synthetic.

### 9.1 Accuracy

No accuracy figures are claimed. The emotion classifier is Prosody Protocol's rule-based heuristic, which abstains when unsure and has been checked only against synthetic clips. There is no sarcasm or urgency classifier, and this repository contains no benchmark dataset. Measured benchmarks on real recordings are on the roadmap (Section 14).

### 9.2 End-to-End Latency (Untested Targets)

| Configuration | STT | LLM | TTS | Total |
|---|---|---|---|---|
| Hybrid | 300ms | 150ms | 200ms | 650ms |
| Local (GPU) | 400ms | 100ms | 300ms | 800ms |
| Local (CPU) | 800ms | 2000ms | 500ms | 3.3s |

These are design targets. Real latency depends on the providers, models, hardware and length of the audio.

### 9.3 Cost

No cost figures are given. Hybrid mode pays per-request fees to its cloud STT (and TTS) vendors, whose prices change; local mode has no per-request fees, but hardware is yours to provide.

---

## 10. Security and Privacy

### 10.1 Data Handling

- **Hybrid:** the cloud STT vendor receives the audio (and a cloud TTS vendor the reply text), under that vendor's own terms and retention; LLM processing stays local when `is_llm_local` is true
- **Local:** all processing on user infrastructure when `is_fully_local` is true
- The engine does not write input audio to disk. Results (transcript, IML, emotion, features) are kept in an in-memory cache until `clear_cache()`, or not at all with `cache_size=0`

### 10.2 Compliance

Intent Engine makes no compliance claim. It has no consent recording, opt-out, retention or deletion features beyond `clear_cache()`, and the IML it writes does not set the `consent` or `processing` attributes of `<iml>`. Running locally keeps audio and transcripts on your infrastructure, which can be a precondition for regimes such as HIPAA or GDPR; meeting one is up to the deployment.

### 10.3 Emotional Data Ethics

Emotional data is treated as sensitive PII. What the code does today:
- Emotion labels and intents are kept out of INFO-and-above logs and out of constitutional `denial_reason`s, except that the TTS adapters log the emotion label they are asked to speak with
- The emotion is optional: nothing downstream requires one, and the pipeline reports none when it cannot tell
- The LLM prompt forbids reading prosody as evidence of lying or truthfulness, judging or profiling the speaker, or basing a consequential decision on prosody alone

Goals the code does not enforce yet: explicit user consent, an opt-in switch for emotional analysis, and review and deletion of a user's emotional metadata.

Project policy: emotional data is never sold, used for manipulation, or used for deception detection.

---

## 11. Accessibility

### 11.1 Atypical Prosody Profiles

Support for users whose prosody does not follow neurotypical patterns (autism, stroke recovery, etc.). A profile is a JSON file (`schemas/prosody-profile.schema.json` in the Prosody Protocol repo) that maps prosodic patterns to the emotion they mean for that speaker:

- Patterns use the schema vocabulary: `pitch`, `pitch_contour`, `volume`, `rate`, `quality`, `pause_frequency` and `emphasis_frequency`; each mapping gives an `emotion` (any label) and a `confidence_boost`. `profile_version` is a semantic version (`X.Y.Z`)
- `IntentEngine(prosody_profile=path)` and `load_profile`/`set_profile` validate the profile and raise `ProfileError` when it is invalid, including the older `f0_mean`/`speech_rate` vocabulary. `create_profile`, `validate_profile` and `clear_profile` complete the API
- The IML assembler applies the profile to each utterance against the speaker's baseline. A matching mapping decides the emotion (the more specific pattern wins), the IML records it as `x-profile`, and `Result.emotion` follows. A mapping shows only when the classifier's confidence plus the boost reaches 0.5, so on a single utterance only a boost of 0.5 or more applies
- A profile can satisfy or dodge an emotion rule of the constitutional filter, so it is trusted configuration, not user input. The `user_id` is not written to the IML or logged at INFO
- Profiles describe speech coming in; they do not affect `type_to_speech`

Baselines are not learned across sessions: the assembler measures the speaker's baseline from the recording it is given.

### 11.2 Augmentative Communication

- `type_to_speech` converts typed text into speech, spoken as typed, with the emotion label shaping the voice as far as the TTS provider allows (Section 7)
- SSML built from the text (predicted pitch and pauses) is used only with a TTS provider that reads SSML; no built-in adapter does

---

## 12. Integration Points

| Platform | Integration Method | Status |
|---|---|---|
| Twilio | Voice webhook handler (`twilio_voice.py`) | Example in `examples/integrations/` |
| Slack | Bot event handler for audio attachments (`slack_bot.py`) | Example in `examples/integrations/` |
| Discord | Bot helper for audio attachments (`discord_bot.py`) | Example in `examples/integrations/` |
| FastAPI REST Server | HTTP endpoints (`/process`, `/generate`, `/synthesize`) (`rest_server.py`) | Example in `examples/integrations/` |
| Anthropic Claude | LLM provider adapter with prosody-aware prompts | Core adapter |
| OpenAI GPT | LLM provider adapter with system prompts | Core adapter |
| Local LLMs | LLM provider adapter (llama.cpp GGUF, or an OpenAI-compatible server such as Ollama or vLLM) with prosody-aware prompts | Core adapter |

The examples are not part of the installed package. From a checkout they import as `examples.integrations.<module>`, and the `examples` extra installs their dependencies.

---

## 13. Dependencies

### 13.1 Core Dependency: Prosody Protocol

| Package | Import | Repository | Purpose |
|---|---|---|---|
| `prosody-protocol[audio]>=0.1.0a3` | `prosody_protocol` | [kase1111-hash/Prosody-Protocol](https://github.com/kase1111-hash/Prosody-Protocol) | IML specification, parser, validator, prosody analysis, emotion classification, accessibility profiles, datasets, benchmarking |

This is a **required** dependency. Intent Engine uses the Prosody Protocol SDK for all IML handling, prosody analysis, and emotion classification. It is not published to PyPI: install it from GitHub with the `audio` extra (see the README). Intent Engine is tested against 0.1.0a3, commit `4d4f0bb930b33f5d66015f8565a87e16c02e5fe2`, which writes IML `0.1.0`; the example REST server also needs its `api` extra. See CLAUDE.md for the full compatibility contract.

Key classes consumed by Intent Engine:
- `IMLParser`, `IMLValidator`, `IMLAssembler` -- IML document lifecycle
- `IMLDocument`, `Utterance`, `Prosody`, `Pause`, `Emphasis`, `Segment` -- Data models
- `ProsodyAnalyzer`, `SpanFeatures`, `WordAlignment`, `PauseInterval` -- Audio analysis
- `prosody_protocol.alignment` (`from_whisper`, `from_deepgram`, `from_assemblyai`) -- Word timings from the STT adapters
- `EmotionClassifier`, `RuleBasedEmotionClassifier` -- Emotion classification
- `IMLToSSML` -- TTS format conversion
- `ProfileLoader`, `ProfileApplier`, `ProsodyProfile` -- Accessibility
- `DatasetLoader`, `DatasetEntry` -- Training data
- `Benchmark`, `BenchmarkReport` -- Evaluation
- `AudioToIML`, `TextToIML`, `IMLToAudio` -- End-to-end converters

### 13.2 External Services / SDKs

Python 3.10 or newer. Each provider is an optional extra (`pyproject.toml` is authoritative):

| Dependency | Extra | Purpose |
|---|---|---|
| `openai-whisper` (and the `ffmpeg` binary) | `whisper` | Base STT model |
| `deepgram-sdk` `>=5,<8` | `deepgram` | Alternative STT |
| `assemblyai` `>=0.20` | `assemblyai` | Alternative STT |
| `anthropic` `>=0.40` | `claude` | LLM provider |
| `openai` `>=1.56` | `openai` | LLM provider, and client for OpenAI-compatible local servers |
| `llama-cpp-python` `>=0.2` | `local-llm` | Local LLM from a GGUF file |
| `elevenlabs` `>=1.8.1` | `elevenlabs` | TTS provider |
| `coqui-tts` `>=0.27` | `coqui` | Open-source TTS (the maintained fork, imported as `TTS`; the original `TTS` 0.22 does not install on Python 3.12 or newer) |
| `pyttsx3` `>=2.99` and the eSpeak NG system library | `espeak` | Open-source TTS |
| `pyyaml` `>=6.0` | (core) | Constitutional rules |
| `fastapi`, `uvicorn`, `python-multipart`, `httpx`, `prosody-protocol[api]`, `twilio`, `slack_sdk`, `discord.py` | `examples` | Example integrations only, not core dependencies |

### 13.3 Related Projects

| Project | Relationship |
|---|---|
| [Prosody Protocol](https://github.com/kase1111-hash/Prosody-Protocol) | IML specification, SDK, and training datasets (core dependency) |
| Mavis | Generates prosody training data via the Prosody Protocol's `MavisBridge` |

---

## 14. Roadmap

### Implemented (Beta -- not yet validated with real audio)
- Core STT + prosody analysis pipeline (provider adapters for Whisper, Deepgram, AssemblyAI)
- LLM integration (Claude, OpenAI, local) with prosody-aware prompts
- Emotional TTS (ElevenLabs, Coqui, eSpeak) with emotion-to-voice parameter mapping (Coqui's is limited to speed)
- Constitutional filter framework (strict YAML rules; fail-closed allow / verify / deny decisions)
- Hybrid and local deployment engines
- Accessibility profiles, applied through the IML assembler

### Next Priority: Core Validation
- End-to-end pipeline validation with real audio files
- Measured accuracy benchmarks on real recordings (emotion detection)
- Constitutional filter demo on real recordings

### Future
- Multi-language support
- Real-time streaming mode
- Expanded emotion granularity
- LLM fine-tuning pipeline for prosody understanding
- Managed cloud service
