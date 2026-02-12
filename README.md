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

---

## How It Works

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: PROSODY-AWARE STT                             │
│  Speech → Text + Emotional Context                      │
│                                                          │
│  Input:  Audio waveform                                 │
│  Output: "I'm <frustrated>fine</frustrated>."           │
│          [IML markup with prosodic features]            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 2: INTENT INTERPRETATION                         │
│  Text + Prosody → Emotional Intent                      │
│                                                          │
│  LLM trained on prosody-protocol dataset                │
│  Understands: sarcasm, urgency, sincerity, etc.         │
│  Constitutional filters: verify genuine intent          │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 3: PROSODY-AWARE TTS                             │
│  Response Text + Emotion → Natural Speech               │
│                                                          │
│  Applies appropriate emotional tone to AI responses     │
│  Matches conversational energy and empathy              │
└─────────────────────────────────────────────────────────┘
```

### Powered By

- **[Prosody Protocol](https://github.com/kase1111-hash/Prosody-Protocol)** - IML specification, SDK, and training datasets
- **Mavis** - Generates prosody training data (via Prosody Protocol's `MavisBridge`)
- **Constitutional AI Principles** - Verifies genuine user intent before actions

---

## Features

### 🎯 Emotional Context Preservation

Captures and interprets:
- **Sarcasm vs. Sincerity** - "Oh great" means different things
- **Urgency Levels** - Fast speech + high pitch = needs immediate help
- **Confidence vs. Uncertainty** - Rising intonation = actually asking, not stating
- **Frustration Markers** - Volume spikes + pitch variation = escalation needed
- **Joy/Enthusiasm** - Positive prosody reinforcement

### 🛡️ Constitutional AI Integration

Verifies intent before executing commands:

```python
# Dangerous command detection
user_speech = "Yeah just delete everything, that'll help"
prosody_analysis = {
    "emotion": "sarcastic",
    "confidence": 0.89,
    "pitch_contour": "fall-rise",  # Classic sarcasm marker
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

### ♿ Accessibility Support

- **Atypical Prosody Profiles** - Learn individual prosodic patterns (autism, stroke, etc.)
- **Custom Emotion Mappings** - "When I speak monotone, I'm excited, not bored"
- **Augmentative Communication** - Turn typed/selected text into emotionally appropriate speech
- **Voice Restoration** - Use prosody from text input for natural-sounding output

### 🔌 Platform Integration

Works with:
- **Anthropic Claude** - Prosody understanding via IML-aware system prompts
- **OpenAI GPT** - Via system prompts and fine-tuning
- **Local Models** - Llama, Mistral with prosody-aware prompts
- **Existing Voice Platforms** - Twilio, Slack, Discord (see `examples/integrations/`)

### 🏠 Deployment Flexibility

- **Hybrid** - STT/TTS in cloud, LLM local
- **Fully Local** - Complete sovereignty (your infrastructure)

---

## Quick Start

### Installation

```bash
pip install intent-engine
```

This installs [Prosody Protocol](https://github.com/kase1111-hash/Prosody-Protocol) (`prosody-protocol`) as a core dependency, which provides IML parsing, validation, prosody analysis, and emotion classification.

### Basic Usage

```python
from intent_engine import IntentEngine

# Initialize with your preferred configuration
engine = IntentEngine(
    stt_provider="whisper-prosody",  # or "deepgram", "assemblyai"
    llm_provider="claude",           # or "openai", "local-llama"
    tts_provider="elevenlabs"        # or "coqui", "espeak"
)

# Process a voice conversation turn
audio_file = "customer_complaint.wav"

# Analyze with prosody
result = engine.process_voice_input(audio_file)

print(f"Transcription: {result.text}")
print(f"Detected emotion: {result.emotion} ({result.confidence:.2%})")
print(f"IML markup: {result.iml}")
print(f"Recommended response tone: {result.suggested_tone}")

# Generate emotionally appropriate response
response = engine.generate_response(
    result.iml,
    context="customer_support",
    tone=result.suggested_tone
)

# Synthesize with appropriate emotion
audio_response = engine.synthesize_speech(
    response.text,
    emotion=response.emotion
)

audio_response.save("response.wav")
```

### With Constitutional Governance

```python
from intent_engine import IntentEngine, ConstitutionalFilter

engine = IntentEngine(llm_provider="claude")

# Define constitutional rules
constitution = ConstitutionalFilter(rules={
    "destructive_actions": {
        "required_prosody": ["calm", "deliberate"],
        "forbidden_prosody": ["sarcastic", "frustrated", "rushed"],
        "verification_level": "explicit_confirmation"
    }
})

# User speaks
audio = "delete_all_files.wav"
result = engine.process_voice_input(audio)

# Check constitutional compliance
if result.intent == "delete_files":
    decision = constitution.evaluate(
        intent=result.intent,
        prosody=result.prosody_features
    )
    
    if decision.allow:
        execute_deletion()
    else:
        respond(f"I detected {result.emotion} in your voice. " +
                f"For safety, I need explicit calm confirmation. " +
                f"Do you really want to delete all files?")
```

---

## Use Cases

### 1. Customer Support

**Before Intent Engine:**
```
Customer: "I've been on hold for 20 minutes!" [angry]
AI: "Thank you for your patience! How may I assist you?"
Customer: [angrier]
```

**With Intent Engine:**
```
Customer: "I've been on hold for 20 minutes!" [angry detected]
AI: "I'm really sorry about the wait - that's unacceptable. 
     Let me get you help immediately." [empathetic tone]
Customer: [feels heard, stays on line]
```

**Potential impact:** Reduced call escalations and improved customer satisfaction by responding with emotional awareness.

### 2. Healthcare

**Mental Health Screening:**
```python
# Detect depression markers from prosody
prosody_features = {
    "pitch_variance": "reduced",  # Flat affect
    "speech_rate": "slow",        # Psychomotor retardation
    "volume": "reduced",          # Low energy
    "pauses": "increased"         # Processing difficulty
}

if depression_risk_score(prosody_features) > 0.7:
    escalate_to_human_clinician()
```

**Elderly Care:**
```python
# Monitor emotional state via daily check-ins
if patient.prosody.shows("increasing_frustration", days=3):
    alert_caregiver("Patient showing signs of distress")
```

### 3. Education

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
# Agent needs permission for sensitive action
agent_request = "delete_old_backups"
user_response = engine.get_voice_confirmation()

if user_response.emotion in ["calm", "confident"]:
    # Genuine authorization
    agent.execute(agent_request)
else:
    # User uncertain or pressured
    agent.defer("I sense hesitation. Let's discuss this first.")
```

### 5. Accessibility

**Non-Verbal Communication:**
```python
# User types or selects text
typed_text = "I'm feeling overwhelmed"

# System adds appropriate prosody for TTS
speech = engine.type_to_speech(
    text=typed_text,
    user_profile="user_123",  # Knows user's prosodic style
    emotion="stressed"
)

# Output sounds genuinely stressed, not robotic
```

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
│         │   • Intent verification            │          │
│         │   • Safety checks                  │          │
│         │   • Governance rules               │          │
│         └────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

### STT Module

**Providers:**

| Provider | Prosody Support | Latency | Cost | Quality |
|----------|----------------|---------|------|---------|
| Whisper + Custom | ✅ Via post-processing | 500ms | $0 | Good |
| Deepgram | ⚠️ Partial (emotions) | 200ms | $0.0125/min | Excellent |
| AssemblyAI | ⚠️ Partial (sentiment) | 300ms | $0.00025/sec | Excellent |
| Custom Fine-tuned | ✅ Full IML output | ~400ms | $0 (after training) | Best (planned) |

**Our Approach:**
- Use base STT for transcription
- Post-process audio with **ProsodyAnalyzer** for IML markup
- Cache results for efficiency

### LLM Module

**Prosody-Aware Prompting:**

```python
SYSTEM_PROMPT = """You are an AI assistant that receives text with IML 
(Intent Markup Language) prosody annotations.

The markup shows HOW words were spoken:
- <prosody pitch="+10%">word</prosody> = higher pitch (excitement, question)
- <emphasis level="strong">word</emphasis> = stressed word (importance)
- <utterance emotion="sarcastic">...</utterance> = overall emotional tone

Always interpret the INTENT behind the prosody, not just literal words.

Examples:
Input: "Oh that's <prosody pitch_contour='fall-rise'>great</prosody>."
Interpretation: Sarcasm detected (pitch contour). User is frustrated.
Response: Acknowledge frustration, ask what's wrong.

Input: "<emphasis>Please</emphasis> listen carefully."
Interpretation: Urgency or frustration. Speaker feels unheard.
Response: Show attentiveness, confirm you're listening.
"""

response = llm.chat(
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": iml_input}]
)
```

### TTS Module

**Emotional Speech Synthesis:**

```python
from intent_engine.tts import create_tts_provider

tts = create_tts_provider("elevenlabs", api_key="...")

# Synthesize with specific emotion
audio = await tts.synthesize(
    text="I understand how you feel.",
    emotion="empathetic"
)
```

### Constitutional Filter

**Rule Definition:**

```yaml
# constitutional_rules.yaml
rules:
  destructive_file_operations:
    triggers:
      - "delete all"
      - "remove everything"
      - "wipe"
    required_prosody:
      emotion: [calm, confident]
      pitch_variance: low
      speaking_rate: [0.8, 1.2]  # Normal pace
    forbidden_prosody:
      emotion: [sarcastic, frustrated, rushed]
    verification:
      method: explicit_confirmation
      retries: 2
      
  financial_transactions:
    triggers:
      - "send money"
      - "transfer"
      - "payment"
    required_prosody:
      emotion: [calm, confident, deliberate]
      pause_before_amount: ">500ms"  # Deliberate consideration
    verification:
      method: two_factor
```

**Runtime Evaluation:**

```python
from intent_engine import ConstitutionalFilter

filter = ConstitutionalFilter.from_yaml("constitutional_rules.yaml")

# Evaluate user request
decision = filter.evaluate(
    intent="delete_all_files",
    prosody_features=prosody_analysis,
    context={"user_id": "123", "session_risk": "low"}
)

if decision.allow:
    execute_action()
elif decision.requires_verification:
    request_explicit_confirmation(method=decision.verification_method)
else:
    deny_with_explanation(reason=decision.denial_reason)
```

---

## Deployment

### Hybrid (Cloud STT/TTS, Local LLM)

```python
from intent_engine import HybridEngine

engine = HybridEngine(
    stt_provider="deepgram",  # Cloud
    llm_provider="local",     # Your GPU
    llm_model="models/llama-3.1-70b-prosody-ft.gguf",
    tts_provider="coqui"      # Local
)
```

**Why Hybrid:**
- STT/TTS benefit from cloud infrastructure (quality)
- LLM runs locally (privacy, sovereignty, no per-request cost)
- Best balance of quality, cost, and control

### Fully Local (Sovereignty Mode)

```python
from intent_engine import LocalEngine

engine = LocalEngine(
    stt_model="whisper-large-v3",
    prosody_model="prosody-analyzer-v2",
    llm_model="llama-3.1-70b-prosody",
    tts_model="coqui-tts-v1"
)

# Everything runs on your infrastructure
result = engine.process("audio.wav")
```

**Hardware Requirements (estimated, not yet benchmarked):**
- **Minimum:** 16GB RAM, CPU-only
- **Recommended:** 32GB RAM, NVIDIA GPU with 24GB+ VRAM
- **Optimal:** 128GB RAM, high-end NVIDIA GPU(s)

> These are estimates based on the underlying model requirements (Whisper, Llama, etc.), not measured benchmarks from Intent Engine itself.

---

## Performance Metrics

> **Note:** The metrics below are design targets, not yet validated by benchmark runs. Actual benchmark infrastructure is planned for a future release.

### Accuracy (Targets)

| Metric | Target | Benchmark |
|--------|--------|-----------|
| Emotion Detection | 87% | Prosody-protocol test set |
| Sarcasm Detection | 82% | SARC dataset |
| Urgency Classification | 91% | Custom healthcare dataset |
| Intent Verification (Constitutional) | 96% | Safety-critical scenarios |

### Latency (Targets)

**End-to-End (voice input → voice response):**

| Configuration | STT | LLM | TTS | Total |
|---------------|-----|-----|-----|-------|
| Hybrid | 300ms | 150ms | 200ms | 650ms |
| Local (GPU) | 400ms | 100ms | 300ms | 800ms |
| Local (CPU) | 800ms | 2000ms | 500ms | 3.3s |

---

## Integration Guides

> **Platform examples:** See `examples/integrations/` for example adapters for Twilio, Slack, Discord, and a FastAPI REST server. These are demonstration code, not part of the core package.

### With Constitutional AI Agents

```python
from intent_engine import IntentEngine, ConstitutionalFilter
from agent_os import Agent

engine = IntentEngine()
constitution = ConstitutionalFilter.from_yaml("rules.yaml")

class ConstitutionalAgent(Agent):
    async def execute_command(self, voice_input):
        # Parse intent from voice
        result = engine.process_voice_input(voice_input)
        
        # Constitutional verification
        decision = constitution.evaluate(
            intent=result.intent,
            prosody_features=result.prosody_features
        )
        
        if decision.allow:
            return await self.execute(result.command)
        elif decision.requires_verification:
            # Ask for explicit confirmation
            confirmation = await self.request_confirmation(
                method=decision.verification_method,
                reason=f"Detected {result.emotion}, need explicit confirmation"
            )
            
            if confirmation.approved and confirmation.prosody.is_calm:
                return await self.execute(result.command)
        
        return self.deny(reason=decision.denial_reason)
```

---

## Security & Privacy

### Data Handling

**Local/Hybrid Mode:**
- All processing on your infrastructure (fully local) or with cloud STT/TTS only (hybrid)
- No permanent audio storage
- Full sovereignty with local deployment

### Compliance

- **HIPAA Ready:** Local deployment option for healthcare
- **GDPR Compliant:** Data minimization, right to deletion

### Emotional Data Ethics

We treat emotional data as **sensitive PII**:
- ✅ Explicit user consent required
- ✅ Opt-in emotional analysis (can be disabled)
- ✅ Users can review/delete emotional metadata
- ❌ Never sell emotional data
- ❌ Never use for manipulation
- ❌ Never use for deception detection

---

## Roadmap

### Implemented (Beta -- not yet validated with real audio)
- [x] Core STT + prosody analysis pipeline (adapters complete, all tests use mocked providers)
- [x] LLM integration (Claude, OpenAI, local) with prosody-aware prompts
- [x] TTS with emotion-to-voice parameter mapping
- [x] Constitutional filter framework (YAML rules, prosody-based verification)
- [x] Hybrid and local deployment engines
- [x] Accessibility profile support (via Prosody Protocol)

### Next Priority: Core Validation
- [ ] End-to-end pipeline validation with real audio files
- [ ] Measured accuracy benchmarks (emotion detection, sarcasm, urgency)
- [ ] Constitutional filter demo with real sarcasm/frustration detection

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
- **IML Parser & Validator** - Parse and validate IML documents against the spec (rules V1-V18)
- **Prosody Analyzer** - Extract F0, intensity, jitter, shimmer, HNR from audio using Praat
- **Emotion Classifier** - Rule-based and ML-based emotion classification from prosodic features
- **IML Assembler** - Build IML documents from STT word alignments and prosody features
- **IML-to-SSML Converter** - Convert IML to SSML for TTS engines
- **Accessibility Profiles** - Load and apply atypical prosody profiles for inclusive design
- **Dataset Tools** - Load, validate, and benchmark prosody-emotion training datasets
- **Mavis Bridge** - Convert data from the Mavis vocal typing game into training datasets

Intent Engine does **not** reimplement any of these components. Instead, it provides the orchestration layer (STT/LLM/TTS provider adapters, constitutional filter, deployment engines) that wires Prosody Protocol's tools into a complete voice AI pipeline.

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
