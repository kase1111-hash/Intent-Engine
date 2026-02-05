# Intent Engine - Execution Guide

A phase-by-phase plan for implementing the Intent Engine from spec to working software.

Each phase is designed to be self-contained and testable before moving to the next. Dependencies flow downward: later phases build on earlier ones.

---

## Phase 0: Project Scaffolding

**Goal:** Establish the Python package structure, build system, development tooling, and CI pipeline so that every subsequent phase has a place to land.

### 0.1 Package Layout

```
intent_engine/
├── __init__.py              # Public API exports
├── engine.py                # IntentEngine orchestrator
├── cloud_engine.py          # CloudEngine
├── hybrid_engine.py         # HybridEngine
├── local_engine.py          # LocalEngine
├── models/
│   ├── __init__.py
│   ├── result.py            # Result dataclass (process_voice_input output)
│   ├── response.py          # Response dataclass (generate_response output)
│   ├── audio.py             # Audio wrapper (synthesize_speech output)
│   └── decision.py          # Constitutional filter Decision dataclass
├── iml/
│   ├── __init__.py
│   ├── parser.py            # IML XML parser
│   ├── builder.py           # IML XML builder
│   └── schema.py            # IML element/attribute definitions
├── prosody/
│   ├── __init__.py
│   ├── analyzer.py          # ProsodyAnalyzer (pitch, energy, tempo)
│   ├── features.py          # ProsodyFeatures dataclass
│   └── emotions.py          # Emotion enum and classification
├── stt/
│   ├── __init__.py
│   ├── base.py              # Abstract STT provider interface
│   ├── whisper.py           # Whisper + post-processing adapter
│   ├── deepgram.py          # Deepgram adapter
│   └── assemblyai.py        # AssemblyAI adapter
├── llm/
│   ├── __init__.py
│   ├── base.py              # Abstract LLM provider interface
│   ├── claude.py            # Anthropic Claude adapter
│   ├── openai.py            # OpenAI adapter
│   ├── local.py             # Local LLM adapter (llama.cpp / vLLM)
│   └── prompts.py           # Prosody-aware system prompts
├── tts/
│   ├── __init__.py
│   ├── base.py              # Abstract TTS provider interface
│   ├── elevenlabs.py        # ElevenLabs adapter
│   ├── coqui.py             # Coqui TTS adapter
│   └── espeak.py            # eSpeak adapter
├── constitutional/
│   ├── __init__.py
│   ├── filter.py            # ConstitutionalFilter class
│   ├── rules.py             # Rule parser (YAML schema)
│   └── evaluator.py         # Prosody-based rule evaluation logic
└── accessibility/
    ├── __init__.py
    ├── profiles.py           # Atypical prosody profile management
    └── augmentative.py       # Typed-text-to-emotional-speech
```

### 0.2 Build System

```
pyproject.toml               # PEP 621 metadata, dependencies, entry points
setup.cfg                    # (optional, for backwards compat)
```

Key decisions:
- Use `pyproject.toml` with `setuptools` or `hatchling` as build backend.
- Define dependency groups: `core`, `whisper`, `deepgram`, `assemblyai`, `elevenlabs`, `coqui`, `espeak`, `dev`, `test`.
- Package name: `intent-engine`. Import name: `intent_engine`.

### 0.3 Development Tooling

| Tool | Purpose |
|------|---------|
| `pytest` | Test runner |
| `pytest-cov` | Coverage reporting |
| `pytest-asyncio` | Async test support |
| `mypy` | Static type checking |
| `ruff` | Linting and formatting |
| `pre-commit` | Git hook management |

### 0.4 CI Pipeline

Set up GitHub Actions:
- **On every push:** lint (`ruff`), type-check (`mypy`), run unit tests (`pytest`).
- **On PR merge to main:** publish to PyPI (or test PyPI).

### 0.5 Deliverables Checklist

- [ ] `pyproject.toml` with all dependency groups
- [ ] Empty module files with `__init__.py` stubs
- [ ] `tests/` directory mirroring `intent_engine/` structure
- [ ] `.github/workflows/ci.yml`
- [ ] `pre-commit` config (`.pre-commit-config.yaml`)
- [ ] `ruff.toml` or `[tool.ruff]` in `pyproject.toml`
- [ ] `mypy.ini` or `[tool.mypy]` in `pyproject.toml`
- [ ] `Makefile` or `justfile` with common commands (`make test`, `make lint`, etc.)

---

## Phase 1: Core Data Models and IML

**Goal:** Define the data structures that flow through the pipeline and the IML parser/builder. Everything downstream depends on these types being stable.

### 1.1 Prosody Features

```python
# intent_engine/prosody/features.py

@dataclass
class ProsodyFeatures:
    pitch: float | None           # % shift from baseline (e.g., +10.0)
    pitch_contour: str | None     # "rise" | "fall" | "fall-rise" | "rise-fall"
    pitch_variance: str | None    # "low" | "normal" | "high" | "reduced"
    volume: float | None          # dB offset (e.g., +6.0)
    speaking_rate: float | None   # Multiplier, 1.0 = normal
    pauses: str | None            # "normal" | "increased" | "decreased"
    emotion: str | None           # Classified emotion string
    confidence: float | None      # 0.0 - 1.0
```

### 1.2 Emotion Taxonomy

```python
# intent_engine/prosody/emotions.py

class Emotion(str, Enum):
    CALM = "calm"
    CONFIDENT = "confident"
    DELIBERATE = "deliberate"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    SARCASTIC = "sarcastic"
    HAPPY = "happy"
    ENTHUSIASTIC = "enthusiastic"
    EXCITED = "excited"
    SAD = "sad"
    STRESSED = "stressed"
    OVERWHELMED = "overwhelmed"
    UNCERTAIN = "uncertain"
    HESITANT = "hesitant"
    RUSHED = "rushed"
    EMPATHETIC = "empathetic"
    NEUTRAL = "neutral"
```

### 1.3 Pipeline Data Objects

Define `Result`, `Response`, `Audio`, and `Decision` dataclasses matching the spec (Section 4.2 and 4.3).

### 1.4 IML Parser and Builder

- **Parser** (`iml/parser.py`): Takes an IML XML string and produces a structured tree of `Utterance`, `ProsodySpan`, and `EmphasisSpan` nodes. Use Python's `xml.etree.ElementTree` for parsing.
- **Builder** (`iml/builder.py`): Takes prosody features + transcription text and produces a valid IML XML string. This is called after STT + prosody analysis to assemble the IML output.
- **Schema** (`iml/schema.py`): Define valid element names, attribute names, attribute value types, and validation rules.

### 1.5 Testing Strategy

- Unit tests for every dataclass (construction, serialization, edge cases).
- IML round-trip tests: build IML string from data, parse it back, assert equality.
- Fuzz testing for IML parser against malformed XML.

### 1.6 Deliverables Checklist

- [ ] `ProsodyFeatures` dataclass with validation
- [ ] `Emotion` enum with all v0.8 values
- [ ] `Result`, `Response`, `Audio`, `Decision` dataclasses
- [ ] IML parser with full element/attribute support
- [ ] IML builder that produces valid markup
- [ ] IML schema validation
- [ ] Unit tests for all of the above (target: 100% coverage for this phase)

---

## Phase 2: Prosody Analyzer

**Goal:** Build the component that extracts pitch, energy, tempo, and other acoustic features from raw audio and classifies the speaker's emotional state. This is the heart of the system -- it turns audio into the prosodic metadata that everything else consumes.

### 2.1 Audio Preprocessing

- Accept WAV, MP3, and other common formats.
- Normalize sample rate (e.g., to 16kHz mono) using `librosa` or `soundfile`.
- Segment audio into utterance-level chunks if needed.

### 2.2 Feature Extraction

| Feature | Method |
|---------|--------|
| **Pitch (F0)** | `librosa.pyin` or `parselmouth` (Praat wrapper). Extract fundamental frequency contour. |
| **Pitch contour** | Derive from F0 trajectory: classify as rise, fall, fall-rise, rise-fall. |
| **Pitch variance** | Standard deviation of F0 over the utterance. Bucket into low/normal/high/reduced. |
| **Volume (energy)** | RMS energy per frame. Compute dB offset from speaker baseline. |
| **Speaking rate** | Syllable count / duration. Or use voice activity detection (VAD) to measure speech-to-pause ratio. |
| **Pauses** | Detect silence intervals > threshold (e.g., 300ms). Classify frequency as normal/increased/decreased. |

### 2.3 Emotion Classification

Two-stage approach:

1. **Rule-based heuristics** (initial implementation): Map feature combinations to emotions using thresholds (e.g., high pitch variance + fast rate + high volume = frustrated).
2. **ML classifier** (upgrade path): Train a small model (SVM, random forest, or lightweight neural net) on labeled prosody-emotion data from the Mavis corpus.

Output: `ProsodyFeatures` object with `emotion` and `confidence` fields populated.

### 2.4 Speaker Baseline

- Track per-speaker baseline statistics (mean pitch, typical volume, normal rate).
- Compare current utterance features against baseline to detect deviations.
- Store baselines in a simple key-value store (dict in memory, or SQLite for persistence).

### 2.5 Testing Strategy

- Use a small set of curated audio files with known emotional content.
- Assert feature extraction outputs are within expected ranges.
- Assert emotion classification matches labeled ground truth for test set.
- Benchmark processing latency.

### 2.6 Deliverables Checklist

- [ ] Audio loader with format normalization
- [ ] Pitch extraction (F0 contour)
- [ ] Pitch contour classification
- [ ] Volume/energy extraction
- [ ] Speaking rate estimation
- [ ] Pause detection
- [ ] Rule-based emotion classifier
- [ ] Speaker baseline tracking
- [ ] `ProsodyAnalyzer` class with a `analyze(audio) -> ProsodyFeatures` method
- [ ] Unit/integration tests with test audio files
- [ ] Latency benchmarks

---

## Phase 3: STT Module (Provider Adapters)

**Goal:** Build the speech-to-text layer with a provider-agnostic interface. Each adapter transcribes audio and then integrates with the Prosody Analyzer to produce IML-annotated output.

### 3.1 Abstract Interface

```python
# intent_engine/stt/base.py

class STTProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio to text.

        Returns:
            TranscriptionResult with .text (str) and .word_timestamps (list)
        """
        ...
```

All adapters implement this interface. The orchestrator calls `transcribe()` then feeds the audio + transcription into the `ProsodyAnalyzer` to produce IML.

### 3.2 Whisper Adapter

- Use OpenAI's Whisper model (via `openai-whisper` package or `faster-whisper`).
- Extract word-level timestamps for alignment with prosody features.
- Run locally -- no API key needed.

### 3.3 Deepgram Adapter

- Use Deepgram's SDK (`deepgram-sdk`).
- Extract word-level timestamps and any available emotion/sentiment metadata.
- Requires API key configuration.

### 3.4 AssemblyAI Adapter

- Use AssemblyAI SDK.
- Extract word-level timestamps and sentiment analysis.
- Requires API key configuration.

### 3.5 Provider Configuration

```python
STT_PROVIDERS = {
    "whisper-prosody": WhisperSTT,
    "deepgram": DeepgramSTT,
    "assemblyai": AssemblyAISTT,
}
```

Providers are selected by string key at `IntentEngine` construction time.

### 3.6 Testing Strategy

- Mock external APIs (Deepgram, AssemblyAI) for unit tests.
- Integration test with Whisper using a short audio clip.
- Verify IML output is valid and features are populated.

### 3.7 Deliverables Checklist

- [ ] `STTProvider` abstract base class
- [ ] `TranscriptionResult` dataclass (text + word timestamps)
- [ ] Whisper adapter (local model)
- [ ] Deepgram adapter (API-based)
- [ ] AssemblyAI adapter (API-based)
- [ ] Provider registry/factory
- [ ] Integration with ProsodyAnalyzer (transcribe then analyze)
- [ ] Unit tests with mocked APIs
- [ ] Integration test with real Whisper model

---

## Phase 4: LLM Module (Intent Interpretation)

**Goal:** Build the layer that receives IML-annotated text and produces an intent interpretation and suggested response. This is where the system understands what the user *meant*, not just what they said.

### 4.1 Abstract Interface

```python
# intent_engine/llm/base.py

class LLMProvider(ABC):
    @abstractmethod
    async def interpret(
        self, iml_input: str, context: str | None = None
    ) -> InterpretationResult:
        """Interpret IML-annotated input and generate a response.

        Returns:
            InterpretationResult with .intent, .response_text, .suggested_emotion
        """
        ...
```

### 4.2 Prosody-Aware System Prompt

Define in `llm/prompts.py`:

- Teach the LLM what IML tags mean.
- Provide examples mapping prosody patterns to intent (e.g., fall-rise pitch contour = sarcasm).
- Instruct the LLM to output structured JSON with `intent`, `response_text`, and `suggested_emotion` fields.
- Version the prompt so it can evolve with the IML spec.

### 4.3 Claude Adapter

- Use the `anthropic` Python SDK.
- Send system prompt + IML-annotated user message.
- Parse structured response.

### 4.4 OpenAI Adapter

- Use the `openai` Python SDK.
- Same system prompt strategy.
- Support both chat completions and function-calling modes.

### 4.5 Local LLM Adapter

- Support loading GGUF models via `llama-cpp-python` or connecting to a local vLLM/Ollama server.
- Same prompt strategy, but may need prompt format adjustments (ChatML, Llama format, etc.).

### 4.6 Testing Strategy

- Mock all LLM API calls in unit tests.
- Provide canned IML inputs and assert intent classification is correct.
- Test prompt rendering for each adapter.
- Integration test with at least one live provider (gated behind an env var flag).

### 4.7 Deliverables Checklist

- [ ] `LLMProvider` abstract base class
- [ ] `InterpretationResult` dataclass
- [ ] Prosody-aware system prompt template
- [ ] Claude adapter
- [ ] OpenAI adapter
- [ ] Local LLM adapter (llama-cpp or vLLM)
- [ ] Provider registry/factory
- [ ] Unit tests with mocked APIs
- [ ] Prompt version tracking

---

## Phase 5: TTS Module (Emotional Speech Synthesis)

**Goal:** Build the text-to-speech layer that takes response text plus an emotion label and produces naturally spoken audio with appropriate emotional tone.

### 5.1 Abstract Interface

```python
# intent_engine/tts/base.py

class TTSProvider(ABC):
    @abstractmethod
    async def synthesize(
        self, text: str, emotion: str, **kwargs
    ) -> AudioData:
        """Synthesize speech with emotional tone.

        kwargs may include pitch_shift, speaking_rate, etc.
        """
        ...
```

### 5.2 ElevenLabs Adapter

- Use ElevenLabs API.
- Map emotion labels to ElevenLabs voice settings (stability, similarity boost, style).
- Support voice selection.

### 5.3 Coqui Adapter

- Use Coqui TTS (open-source, runs locally).
- Map emotion labels to speaker embeddings or style tokens.
- Useful for hybrid/local deployments.

### 5.4 eSpeak Adapter

- Use eSpeak (lightweight, open-source).
- Map emotion to SSML parameters (pitch, rate, volume).
- Lowest quality but runs anywhere, useful for testing and accessibility fallback.

### 5.5 Emotion-to-Voice Parameter Mapping

Define a mapping table:

| Emotion | Pitch Shift | Rate | Volume | Style Notes |
|---------|-------------|------|--------|-------------|
| empathetic | -5% | 0.9x | -2dB | Warm, slightly slower |
| frustrated | +5% | 1.1x | +3dB | Tense, slightly faster |
| calm | 0% | 0.95x | 0dB | Even, measured |
| enthusiastic | +10% | 1.15x | +2dB | Bright, upbeat |

Each adapter translates this table into provider-specific parameters.

### 5.6 Audio Output

The `Audio` object should support:
- `.save(path)` -- write to file (WAV, MP3)
- `.bytes` -- raw audio bytes
- `.url` -- for cloud-hosted results (ElevenLabs returns a URL)
- `.duration` -- audio duration in seconds

### 5.7 Deliverables Checklist

- [ ] `TTSProvider` abstract base class
- [ ] `AudioData` class with save/bytes/url/duration
- [ ] ElevenLabs adapter
- [ ] Coqui adapter
- [ ] eSpeak adapter
- [ ] Emotion-to-parameter mapping table
- [ ] Provider registry/factory
- [ ] Unit tests with mocked APIs
- [ ] Audio output validation (correct format, non-empty)

---

## Phase 6: Constitutional Filter

**Goal:** Build the safety system that evaluates user intent against prosodic features before allowing sensitive actions. This is the governance layer.

### 6.1 YAML Rule Parser

Parse constitutional rules from YAML following the schema in spec Section 4.4:

```yaml
rules:
  rule_name:
    triggers: [...]
    required_prosody:
      emotion: [...]
      pitch_variance: ...
      speaking_rate: [min, max]
    forbidden_prosody:
      emotion: [...]
    verification:
      method: explicit_confirmation | two_factor
      retries: N
```

Validate the YAML against a JSON Schema or use Pydantic models for parsing.

### 6.2 Trigger Matching

- Match user intent string (from the LLM module) against rule triggers.
- Support exact match and keyword-in-intent matching.
- Return all matching rules (multiple rules may apply to a single intent).

### 6.3 Prosody Evaluation

For each matching rule:
1. Check `required_prosody`: all conditions must be met. If the user's emotion is not in the allowed list, or speaking rate is outside the allowed range, the check fails.
2. Check `forbidden_prosody`: if any forbidden condition matches, the check fails.
3. If the check fails, determine the verification method and build a `Decision` object.

### 6.4 Decision Logic

```
IF no rules match intent → Decision(allow=True)
IF rules match AND prosody passes all checks → Decision(allow=True)
IF rules match AND prosody fails checks AND verification defined → Decision(
    allow=False, requires_verification=True, verification_method=..., denial_reason=...
)
IF rules match AND prosody fails AND no verification fallback → Decision(
    allow=False, requires_verification=False, denial_reason=...
)
```

### 6.5 Testing Strategy

- Test every branch of the decision logic.
- Test with various prosody feature combinations against sample rules.
- Test YAML parsing with valid and invalid rule files.
- Test edge cases: empty rules, overlapping rules, missing fields.

### 6.6 Deliverables Checklist

- [ ] YAML rule schema definition (Pydantic model or JSON Schema)
- [ ] Rule parser (`ConstitutionalFilter.from_yaml(path)`)
- [ ] Trigger matching engine
- [ ] Prosody evaluation logic (required + forbidden checks)
- [ ] `Decision` object construction
- [ ] `ConstitutionalFilter.evaluate(intent, prosody_features, context)` method
- [ ] Unit tests covering all decision branches
- [ ] Sample `constitutional_rules.yaml` for testing

---

## Phase 7: IntentEngine Orchestrator

**Goal:** Wire everything together. The `IntentEngine` class coordinates the full pipeline: audio in, IML out, response generated, speech synthesized.

### 7.1 Pipeline Flow

```
process_voice_input(audio_file):
    1. STT provider transcribes audio → text + word timestamps
    2. ProsodyAnalyzer extracts features from audio → ProsodyFeatures
    3. IML builder combines text + features → IML string
    4. Return Result(text, emotion, confidence, iml, suggested_tone, prosody_features, intent)

generate_response(iml, context, tone):
    1. LLM provider interprets IML → intent + response text + emotion
    2. Return Response(text, emotion)

synthesize_speech(text, emotion):
    1. TTS provider synthesizes text with emotion → AudioData
    2. Return Audio object
```

### 7.2 Constructor

```python
class IntentEngine:
    def __init__(
        self,
        stt_provider: str = "whisper-prosody",
        llm_provider: str = "claude",
        tts_provider: str = "elevenlabs",
        constitutional_rules: str | None = None,
        **provider_kwargs
    ):
        self._stt = STT_PROVIDERS[stt_provider](**provider_kwargs)
        self._llm = LLM_PROVIDERS[llm_provider](**provider_kwargs)
        self._tts = TTS_PROVIDERS[tts_provider](**provider_kwargs)
        self._prosody = ProsodyAnalyzer()
        self._iml_builder = IMLBuilder()
        self._filter = (
            ConstitutionalFilter.from_yaml(constitutional_rules)
            if constitutional_rules
            else None
        )
```

### 7.3 Error Handling

- Wrap provider errors in `IntentEngineError` subclasses: `STTError`, `LLMError`, `TTSError`.
- If prosody analysis fails, fall back to text-only mode (IML without prosody tags) and log a warning.
- If the constitutional filter is loaded and a sensitive intent is detected, always route through the filter before returning.

### 7.4 Caching

- Cache STT results keyed by audio file hash (avoid re-transcribing).
- Cache prosody analysis results alongside STT results.
- Use an in-memory LRU cache initially; make the cache backend pluggable later.

### 7.5 Async Support

All public methods should be `async`. Provide synchronous wrappers for convenience:

```python
def process_voice_input_sync(self, audio_file: str) -> Result:
    return asyncio.run(self.process_voice_input(audio_file))
```

### 7.6 Deliverables Checklist

- [ ] `IntentEngine` class with constructor and provider wiring
- [ ] `process_voice_input()` full pipeline
- [ ] `generate_response()` method
- [ ] `synthesize_speech()` method
- [ ] Error hierarchy (`IntentEngineError`, `STTError`, `LLMError`, `TTSError`)
- [ ] Graceful fallback when prosody analysis fails
- [ ] Result caching (LRU, keyed by audio hash)
- [ ] Async methods with sync wrappers
- [ ] End-to-end integration tests (mocked providers)
- [ ] End-to-end integration test (real Whisper + mocked LLM + eSpeak)

---

## Phase 8: Deployment Engines

**Goal:** Implement the three deployment-specific engine variants that wrap `IntentEngine` with appropriate defaults and constraints.

### 8.1 CloudEngine

```python
class CloudEngine:
    def __init__(self, api_key: str):
        # Validates API key against Intent Engine cloud service
        # All processing happens server-side via API calls
        ...
```

- Wraps a REST client that calls the Intent Engine managed service.
- Handles authentication, retries, rate limiting.
- Returns the same `Result`, `Response`, `Audio` objects as `IntentEngine`.

### 8.2 HybridEngine

```python
class HybridEngine:
    def __init__(
        self,
        stt_provider: str = "deepgram",
        llm_provider: str = "local",
        llm_model: str = "...",
        tts_provider: str = "coqui",
    ):
        # Cloud STT/TTS, local LLM
        ...
```

- Validates that the LLM provider is configured for local execution.
- Uses cloud STT and TTS for quality, local LLM for privacy.
- Inherits from or delegates to `IntentEngine`.

### 8.3 LocalEngine

```python
class LocalEngine:
    def __init__(
        self,
        stt_model: str,
        prosody_model: str,
        llm_model: str,
        tts_model: str,
    ):
        # Everything runs locally
        ...
```

- Validates that all models are available on disk.
- No network calls.
- Reports hardware capabilities and warns if under minimum requirements.

### 8.4 Deliverables Checklist

- [ ] `CloudEngine` with API client and auth
- [ ] `HybridEngine` with mixed cloud/local configuration
- [ ] `LocalEngine` with full local processing
- [ ] Shared interface: all three expose `process_voice_input`, `generate_response`, `synthesize_speech`
- [ ] Configuration validation (e.g., LocalEngine rejects cloud providers)
- [ ] Unit tests for each engine variant

---

## Phase 9: Integrations and Platform Adapters

**Goal:** Provide ready-made adapters for common voice platforms so users can drop Intent Engine into existing infrastructure.

### 9.1 Twilio Integration

- Flask/FastAPI webhook handler that receives Twilio voice callbacks.
- Processes recording URL through Intent Engine.
- Returns TwiML with synthesized audio response.
- Package as an importable helper: `from intent_engine.integrations.twilio import TwilioVoiceHandler`.

### 9.2 Slack/Discord Bot Helpers

- Utility functions for recording voice channel audio and processing through the engine.
- Event handler decorators or callback patterns.

### 9.3 Generic Webhook Server

- Standalone FastAPI server that exposes Intent Engine as a REST API.
- Endpoints: `POST /process` (audio upload), `POST /generate` (IML input), `POST /synthesize` (text + emotion).
- Useful as a microservice deployment.

### 9.4 Deliverables Checklist

- [ ] Twilio voice webhook handler
- [ ] Slack bot helper
- [ ] Discord bot helper
- [ ] Generic REST API server (`intent_engine.server`)
- [ ] Integration examples in `examples/` directory
- [ ] Documentation for each integration

---

## Phase 10: Accessibility Features

**Goal:** Implement support for atypical prosody profiles and augmentative communication, ensuring the system works for users whose vocal patterns differ from neurotypical norms.

### 10.1 Atypical Prosody Profiles

- `ProsodyProfile` dataclass that stores a user's baseline prosody statistics and custom emotion mappings.
- Example: a user with autism may have flat affect (low pitch variance) even when excited. Their profile maps `low_pitch_variance + fast_rate` → `excited` instead of the default mapping.
- Profiles are stored as JSON and loaded per-user.

### 10.2 Custom Emotion Mappings

- Allow users to define: "When I do X, I mean Y."
- Override the default emotion classifier with user-specific rules.
- These take priority over the general classifier.

### 10.3 Augmentative Communication (Type-to-Speech)

- `type_to_speech(text, user_profile, emotion)` method.
- Takes typed text, applies the user's prosody profile, and synthesizes speech that sounds like the user *would* sound if they could speak with that emotion.
- Useful for non-verbal users or users with speech impairments.

### 10.4 Deliverables Checklist

- [ ] `ProsodyProfile` dataclass and JSON serialization
- [ ] Profile-aware emotion classification (override defaults)
- [ ] Profile management (create, load, update, delete)
- [ ] `type_to_speech()` method on IntentEngine
- [ ] Unit tests with sample profiles
- [ ] Documentation on creating custom profiles

---

## Phase 11: Testing, Benchmarking, and Quality

**Goal:** Establish comprehensive test coverage, performance benchmarks, and quality gates.

### 11.1 Test Pyramid

| Level | Scope | Tools |
|-------|-------|-------|
| **Unit** | Individual classes and functions | `pytest`, mocks |
| **Integration** | Module interactions (e.g., STT → Prosody → IML) | `pytest`, test fixtures |
| **End-to-End** | Full pipeline with real or mocked providers | `pytest`, test audio files |
| **Contract** | API response shapes match spec | `pydantic` validation |
| **Performance** | Latency and throughput benchmarks | `pytest-benchmark` |

### 11.2 Test Data

- Curate a set of 20-50 short audio clips with labeled emotions.
- Store in `tests/fixtures/audio/`.
- Include diverse speakers, accents, and emotional states.
- Include atypical prosody samples for accessibility testing.

### 11.3 Performance Targets (from spec Section 9)

| Metric | Target |
|--------|--------|
| Emotion detection accuracy | 87% on prosody-protocol test set |
| Sarcasm detection accuracy | 82% on SARC dataset |
| Urgency classification accuracy | 91% on custom healthcare dataset |
| Constitutional intent verification | 96% on safety-critical scenarios |
| End-to-end latency (cloud) | < 1.2s |
| End-to-end latency (hybrid) | < 650ms |
| End-to-end latency (local GPU) | < 800ms |

### 11.4 Quality Gates for CI

- All unit tests pass.
- Type checking passes (`mypy --strict`).
- Linting passes (`ruff check`).
- Test coverage >= 80% (aim for 90%+ on core modules).
- No security vulnerabilities (`pip-audit` or `safety`).

### 11.5 Deliverables Checklist

- [ ] Unit tests for every module (target: 90%+ coverage)
- [ ] Integration test suite
- [ ] End-to-end test with mocked providers
- [ ] Labeled test audio dataset
- [ ] Performance benchmark suite
- [ ] CI quality gates configured
- [ ] `pip-audit` or `safety` in CI

---

## Phase 12: Fine-Tuning Pipeline (Advanced)

**Goal:** Provide tooling for users who want to fine-tune local LLMs on the prosody-protocol dataset for improved intent interpretation.

### 12.1 Scope

This phase is optional for the initial release but is described in the spec (Section 6.2) and README. It enables users to train models that natively understand IML rather than relying on prompt engineering.

### 12.2 Components

- **Data loader**: Load the `prosody-protocol/mavis-corpus` dataset.
- **FineTuner class**: Configure and run fine-tuning jobs.
- **Evaluation**: Benchmark fine-tuned model against prompt-based baseline.

### 12.3 Deliverables Checklist

- [ ] Dataset loader for Mavis corpus
- [ ] `FineTuner` class with configurable hyperparameters
- [ ] Training script (`intent_engine.training`)
- [ ] Evaluation script comparing fine-tuned vs. prompt-based accuracy
- [ ] Documentation on running fine-tuning

---

## Phase Dependency Graph

```
Phase 0: Scaffolding
    │
    ▼
Phase 1: Data Models & IML
    │
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
Phase 2:       Phase 4:       Phase 5:
Prosody        LLM Module     TTS Module
Analyzer           │              │
    │              │              │
    ▼              │              │
Phase 3:           │              │
STT Module         │              │
    │              │              │
    └──────┬───────┘              │
           ▼                      │
       Phase 6:                   │
       Constitutional             │
       Filter                     │
           │                      │
           └──────┬───────────────┘
                  ▼
              Phase 7:
              IntentEngine
              Orchestrator
                  │
        ┌─────────┼──────────┐
        ▼         ▼          ▼
    Phase 8:  Phase 9:   Phase 10:
    Deployment Integrations Accessibility
    Engines
        │         │          │
        └─────────┼──────────┘
                  ▼
              Phase 11:
              Testing &
              Benchmarking
                  │
                  ▼
              Phase 12:
              Fine-Tuning
              (Optional)
```

**Key parallelism opportunities:**
- Phases 2, 4, and 5 can be developed in parallel after Phase 1 is complete.
- Phases 8, 9, and 10 can be developed in parallel after Phase 7 is complete.

---

## Recommended Execution Order

For a solo developer or small team, the recommended sequential order is:

| Order | Phase | Rationale |
|-------|-------|-----------|
| 1 | Phase 0: Scaffolding | Foundation for everything |
| 2 | Phase 1: Data Models & IML | All modules depend on these types |
| 3 | Phase 2: Prosody Analyzer | Core differentiator; needed before STT integration |
| 4 | Phase 3: STT Module | First visible end-to-end slice (audio → IML) |
| 5 | Phase 4: LLM Module | Second pipeline layer |
| 6 | Phase 5: TTS Module | Third pipeline layer |
| 7 | Phase 6: Constitutional Filter | Safety layer, depends on prosody + LLM output |
| 8 | Phase 7: Orchestrator | Wires everything together |
| 9 | Phase 8: Deployment Engines | Packaging for different deployment modes |
| 10 | Phase 9: Integrations | Platform-specific adapters |
| 11 | Phase 10: Accessibility | User-facing accessibility features |
| 12 | Phase 11: Testing & Benchmarks | Comprehensive quality pass |
| 13 | Phase 12: Fine-Tuning | Advanced / optional |

**Milestone checkpoints:**
- After Phase 3: Demo "audio in → IML out" (prosody-annotated transcription).
- After Phase 7: Demo full pipeline "audio in → spoken response out."
- After Phase 8: All three deployment modes functional.
- After Phase 11: Release candidate.
