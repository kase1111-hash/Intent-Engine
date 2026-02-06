# Intent Engine - Technical Specification

**Version:** 0.8.0 (Beta)
**Author:** Kase Branham
**License:** TBD (Commercial-friendly, patent-grant included, attribution required)

---

## 1. Overview

Intent Engine is a prosody-aware AI system that preserves and interprets emotional intent throughout a voice conversation pipeline. It bridges the gap between what a user says (words) and how they mean it (tone, emphasis, pitch, rhythm), enabling AI systems to respond with appropriate emotional intelligence.

### 1.1 Problem Statement

Current voice AI systems lose 60-80% of emotional context because speech-to-text strips away prosody. This leads to:

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
- Offer flexible deployment: cloud, hybrid, and fully local

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
│             AssemblyAI, Custom       │
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
| **STT Module** | Transcribes audio and extracts raw prosodic features |
| **Prosody Analyzer** | Extracts pitch, energy, tempo; classifies emotion |
| **LLM Module** | Interprets intent using prosody-aware prompts or fine-tuned models |
| **Constitutional Filter** | Verifies genuine intent before executing sensitive actions |
| **TTS Module** | Synthesizes speech with appropriate emotional tone |
| **IntentEngine (Orchestrator)** | Coordinates the full pipeline end-to-end |

---

## 3. Intent Markup Language (IML)

IML is an XML-based markup language that carries prosodic information through the pipeline. The IML specification, XML Schema (XSD), parser, validator, and data models are defined and maintained by the **[Prosody Protocol](https://github.com/kase1111-hash/Prosody-Protocol)** project. Intent Engine consumes IML via the `prosody_protocol` SDK and does not maintain its own IML implementation.

**Canonical references:**
- IML XSD Schema: `schemas/iml-1.0.xsd` in the Prosody Protocol repo
- Parser: `prosody_protocol.IMLParser`
- Validator: `prosody_protocol.IMLValidator` (rules V1-V18)
- Data models: `prosody_protocol.models` (`IMLDocument`, `Utterance`, `Prosody`, `Pause`, `Emphasis`, `Segment`)

### 3.1 Elements

| Element | Purpose | Attributes |
|---|---|---|
| `<iml>` | Root wrapper for multi-utterance documents | `version`, `language`, `consent`, `processing` |
| `<utterance>` | Wraps a full spoken turn | `emotion`, `confidence`, `speaker_id` |
| `<prosody>` | Marks prosodic features on a span | `pitch`, `pitch_contour`, `volume`, `rate`, `quality` + extended attrs |
| `<pause>` | Explicit timing gap | `duration` (ms, required, positive integer) |
| `<emphasis>` | Marks stressed words | `level` (strong, moderate, reduced) |
| `<segment>` | Clause-level prosodic grouping | `tempo`, `rhythm` (direct child of `<utterance>` only) |

### 3.2 Example

```xml
<utterance emotion="frustrated" confidence="0.91">
  This is the <emphasis level="strong">THIRD</emphasis> TIME
  I've called about this!
</utterance>
```

### 3.3 Prosodic Features Captured

**Core attributes (on `<prosody>`):**

| Attribute | Description | Value Format |
|---|---|---|
| `pitch` | Fundamental frequency shift | `+N%`, `-N%`, `+Nst`, `-Nst`, or `NHz` |
| `pitch_contour` | Pitch trajectory pattern | `rise`, `fall`, `rise-fall`, `fall-rise`, `fall-sharp`, `rise-sharp`, `flat` |
| `volume` | Loudness relative to baseline | `+NdB`, `-NdB` |
| `rate` | Speaking rate | `fast`, `slow`, `medium`, or `N%` |
| `quality` | Voice quality | `modal`, `breathy`, `tense`, `creaky`, `whispery`, `harsh` |

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
| `jitter` | float | Voice quality measure |
| `shimmer` | float | Voice quality measure |
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

**Intent Engine extended vocabulary (custom, triggers V15 info notice):**
- `confident`, `deliberate`
- `happy`, `enthusiastic`, `excited`
- `stressed`, `overwhelmed`
- `hesitant`, `rushed`

**Planned (v1.0):** 20+ fine-grained emotions with standardization across both projects.

---

## 4. Public API

### 4.1 Main Entry Points

#### IntentEngine (General Purpose)

```python
from intent_engine import IntentEngine

engine = IntentEngine(
    stt_provider="whisper-prosody",    # "whisper-prosody" | "deepgram" | "assemblyai"
    llm_provider="claude",             # "claude" | "openai" | "local-llama"
    tts_provider="elevenlabs"          # "elevenlabs" | "coqui" | "espeak"
)
```

#### CloudEngine (Managed Service)

```python
from intent_engine import CloudEngine

engine = CloudEngine(api_key="ie_sk_...")
```

#### HybridEngine (Cloud STT/TTS + Local LLM)

```python
from intent_engine import HybridEngine

engine = HybridEngine(
    stt_provider="deepgram",
    llm_provider="local",
    llm_model="models/llama-3.1-70b-prosody-ft.gguf",
    tts_provider="coqui"
)
```

#### LocalEngine (Full Sovereignty)

```python
from intent_engine import LocalEngine

engine = LocalEngine(
    stt_model="whisper-large-v3",
    prosody_model="prosody-analyzer-v2",
    llm_model="llama-3.1-70b-prosody",
    tts_model="coqui-tts-v1"
)
```

### 4.2 Core Methods

#### `process_voice_input(audio_file) -> Result`

Processes an audio file through STT and prosody analysis.

**Parameters:**
- `audio_file` (str): Path or URL to an audio file (WAV, MP3, etc.)

**Returns:** `Result` object with:
| Field | Type | Description |
|---|---|---|
| `text` | `str` | Plain text transcription |
| `emotion` | `str` | Detected primary emotion |
| `confidence` | `float` | Emotion detection confidence (0.0-1.0) |
| `iml` | `str` | Full IML markup of the utterance |
| `suggested_tone` | `str` | Recommended tone for the response |
| `prosody_features` | `dict` | Raw prosodic feature dictionary |
| `intent` | `str` | Parsed user intent |

#### `generate_response(iml, context, tone) -> Response`

Generates an LLM response using IML-annotated input.

**Parameters:**
- `iml` (str): IML-annotated input text
- `context` (str): Conversation context (e.g., `"customer_support"`)
- `tone` (str): Desired response tone

**Returns:** `Response` object with:
| Field | Type | Description |
|---|---|---|
| `text` | `str` | Response text |
| `emotion` | `str` | Emotion to apply in TTS |

#### `synthesize_speech(text, emotion) -> Audio`

Synthesizes speech with the specified emotional tone.

**Parameters:**
- `text` (str): Text to synthesize
- `emotion` (str): Emotional tone to apply

**Returns:** `Audio` object (supports `.save(path)`, `.url`)

### 4.3 Constitutional Filter

```python
from intent_engine import ConstitutionalFilter

filter = ConstitutionalFilter.from_yaml("constitutional_rules.yaml")

decision = filter.evaluate(
    intent="delete_all_files",
    prosody_features=prosody_analysis,
    context={"user_id": "123", "session_risk": "low"}
)
```

**Decision object:**
| Field | Type | Description |
|---|---|---|
| `allow` | `bool` | Whether the action is permitted |
| `requires_verification` | `bool` | Whether additional confirmation is needed |
| `verification_method` | `str` | Method to use (e.g., `"explicit_confirmation"`, `"two_factor"`) |
| `denial_reason` | `str` | Human-readable reason for denial |

### 4.4 Constitutional Rules Schema (YAML)

```yaml
rules:
  <rule_name>:
    triggers:
      - "<keyword or phrase>"
    required_prosody:
      emotion: [<list of acceptable emotions>]
      pitch_variance: <low|normal|high>
      speaking_rate: [<min>, <max>]
    forbidden_prosody:
      emotion: [<list of blocking emotions>]
    verification:
      method: <explicit_confirmation|two_factor>
      retries: <int>
```

---

## 5. STT Provider Specifications

| Provider | Prosody Support | Latency | Cost | Quality |
|---|---|---|---|---|
| Whisper + Custom Post-Processing | Full (via ProsodyAnalyzer) | ~500ms | Free (self-hosted) | Good |
| Deepgram | Partial (emotions only) | ~200ms | $0.0125/min | Excellent |
| AssemblyAI | Partial (sentiment only) | ~300ms | $0.00025/sec | Excellent |
| Custom Fine-Tuned | Full IML output | ~400ms | Free (after training) | Best |

**Processing approach:** Use base STT for transcription, then post-process audio with the ProsodyAnalyzer to generate IML markup. Results are cached for efficiency.

---

## 6. LLM Integration

### 6.1 Prompting Strategy

The LLM receives a system prompt that teaches it to interpret IML annotations. The prompt defines:

- How to read IML markup tags
- How to interpret prosodic features (pitch contours, emphasis, emotion tags)
- How to map prosody to intent (e.g., fall-rise pitch = sarcasm)
- Response guidelines based on detected emotion

### 6.2 Fine-Tuning

For advanced use cases, models can be fine-tuned on the prosody-protocol dataset using:

- Base model: Meta Llama 3.1 70B (or equivalent)
- Dataset: `prosody-protocol/mavis-corpus`
- Task: `prosody_to_intent`

---

## 7. TTS Emotional Synthesis

The TTS module applies emotional parameters to speech output:

| Parameter | Type | Description |
|---|---|---|
| `emotion` | `str` | Target emotion for the synthesized speech |
| `pitch_shift` | `int` | Pitch adjustment in semitones |
| `speaking_rate` | `float` | Rate multiplier (1.0 = normal) |

---

## 8. Deployment Modes

### 8.1 Cloud (Managed Service)

- All processing handled by Intent Engine infrastructure
- API key authentication
- Usage-based pricing:
  - STT: $0.02/minute
  - LLM: $0.001/request
  - TTS: $0.015/1000 characters
  - Free tier: 1000 requests/month

### 8.2 Hybrid

- STT/TTS in cloud (quality benefit)
- LLM runs locally (privacy, no per-request cost)
- Best balance of quality, cost, and control

### 8.3 Fully Local (Sovereignty Mode)

- All processing on user infrastructure
- No data leaves the network

**Hardware Requirements:**
| Tier | RAM | GPU | Performance |
|---|---|---|---|
| Minimum | 16 GB | CPU-only | Slow |
| Recommended | 32 GB | NVIDIA RTX 4090 | Good |
| Optimal | 128 GB | 2x NVIDIA A100 | Best |

---

## 9. Performance Targets

### 9.1 Accuracy

| Metric | Target | Benchmark Dataset |
|---|---|---|
| Emotion Detection | 87% | Prosody-protocol test set |
| Sarcasm Detection | 82% | SARC dataset |
| Urgency Classification | 91% | Custom healthcare dataset |
| Intent Verification (Constitutional) | 96% | Safety-critical scenarios |

### 9.2 End-to-End Latency

| Configuration | STT | LLM | TTS | Total |
|---|---|---|---|---|
| Cloud (All) | 300ms | 400ms | 500ms | 1.2s |
| Hybrid | 300ms | 150ms | 200ms | 650ms |
| Local (GPU) | 400ms | 100ms | 300ms | 800ms |
| Local (CPU) | 800ms | 2000ms | 500ms | 3.3s |

### 9.3 Cost (1000 conversations/month, ~10 turns each)

| Deployment | Monthly Cost |
|---|---|
| Cloud (Managed) | ~$150 |
| Hybrid | ~$40 |
| Local | $0 (hardware amortized) |

---

## 10. Security and Privacy

### 10.1 Data Handling

- **Cloud:** TLS 1.3 encryption in transit; temporary processing only; no permanent audio storage; optional zero-logging mode
- **Local:** All processing on user infrastructure; no data leaves the network

### 10.2 Compliance Targets

- HIPAA Ready (local deployment)
- GDPR Compliant (data minimization, right to deletion)
- SOC 2 Type II (cloud service, in progress)

### 10.3 Emotional Data Ethics

Emotional data is treated as sensitive PII:
- Explicit user consent required
- Opt-in emotional analysis (can be disabled)
- Users can review and delete emotional metadata
- Emotional data is never sold, used for manipulation, or used for deception detection

---

## 11. Accessibility

### 11.1 Atypical Prosody Profiles

Support for users whose prosody does not follow neurotypical patterns (autism, stroke recovery, etc.):
- Custom emotion mappings per user
- Learnable individual prosodic baselines

### 11.2 Augmentative Communication

- Convert typed/selected text into emotionally appropriate speech
- Use prosody context from text input for natural-sounding TTS output

---

## 12. Integration Points

| Platform | Integration Method |
|---|---|
| Twilio | Voice webhook handler |
| Vonage | Voice API adapter |
| Amazon Connect | Contact flow integration |
| Slack | Bot event handler (voice channels) |
| Discord | Voice state update listener |
| Anthropic Claude | Native API with prosody-aware prompts |
| OpenAI GPT | System prompts and fine-tuning |
| Local LLMs (Llama, Mistral) | Prosody-aware prompt templates |

---

## 13. Dependencies

### 13.1 Core Dependency: Prosody Protocol

| Package | Import | Repository | Purpose |
|---|---|---|---|
| `prosody-protocol` | `prosody_protocol` | [kase1111-hash/Prosody-Protocol](https://github.com/kase1111-hash/Prosody-Protocol) | IML specification, parser, validator, prosody analysis, emotion classification, accessibility profiles, datasets, benchmarking |

This is a **required** dependency. Intent Engine uses the Prosody Protocol SDK for all IML handling, prosody analysis, and emotion classification. See CLAUDE.md for the full compatibility contract.

Key classes consumed by Intent Engine:
- `IMLParser`, `IMLValidator`, `IMLAssembler` -- IML document lifecycle
- `IMLDocument`, `Utterance`, `Prosody`, `Pause`, `Emphasis`, `Segment` -- Data models
- `ProsodyAnalyzer`, `SpanFeatures`, `WordAlignment`, `PauseInterval` -- Audio analysis
- `EmotionClassifier`, `RuleBasedEmotionClassifier` -- Emotion classification
- `IMLToSSML` -- TTS format conversion
- `ProfileLoader`, `ProfileApplier`, `ProsodyProfile` -- Accessibility
- `DatasetLoader`, `DatasetEntry` -- Training data
- `Benchmark`, `BenchmarkReport` -- Evaluation
- `AudioToIML`, `TextToIML`, `IMLToAudio` -- End-to-end converters

### 13.2 External Services / SDKs

| Dependency | Purpose |
|---|---|
| OpenAI Whisper | Base STT model |
| Anthropic Claude API | LLM provider |
| Deepgram SDK | Alternative STT |
| AssemblyAI SDK | Alternative STT |
| ElevenLabs API | TTS provider |
| Coqui TTS | Open-source TTS |
| eSpeak | Open-source TTS |
| Twilio SDK | Telephony integration |

### 13.3 Related Projects

| Project | Relationship |
|---|---|
| [Prosody Protocol](https://github.com/kase1111-hash/Prosody-Protocol) | IML specification, SDK, and training datasets (core dependency) |
| [Mavis](https://github.com/yourusername/mavis) | Generates prosody training data via the Prosody Protocol's MavisBridge |

---

## 14. Roadmap

| Quarter | Milestones |
|---|---|
| **Q1 2026 (Beta)** | Core STT + prosody analysis; LLM integration (Claude, OpenAI, local); basic emotional TTS; constitutional filter framework; production cloud service |
| **Q2 2026** | Multi-language support (5 languages); real-time streaming mode; 20+ emotion granularity; custom prosody profile builder |
| **Q3 2026** | Enterprise features (SSO, audit logs, analytics); telephony integrations; Agent-OS deep integration; healthcare fine-tuning |
| **Q4 2026** | Mobile SDKs (iOS, Android); edge deployment (on-device); research API; W3C standardization proposal |
