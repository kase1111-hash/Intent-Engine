# Intent Engine - Execution Guide

A phase-by-phase plan for implementing the Intent Engine from spec to working software.

Each phase is designed to be self-contained and testable before moving to the next. Dependencies flow downward: later phases build on earlier ones.

**Critical dependency:** Intent Engine is built on the **[Prosody Protocol](https://github.com/kase1111-hash/Prosody-Protocol)** SDK (`prosody-protocol` on PyPI, `prosody_protocol` for import). The Prosody Protocol provides IML parsing, validation, data models, prosody analysis, emotion classification, accessibility profiles, and dataset tooling. Intent Engine does NOT reimplement any of these -- it provides the orchestration layer, provider adapters, constitutional filter, and deployment engines on top.

---

## Phase 0: Project Scaffolding

**Goal:** Establish the Python package structure, build system, development tooling, and CI pipeline so that every subsequent phase has a place to land.

### 0.1 Package Layout

```
intent_engine/
├── __init__.py              # Public API exports
├── engine.py                # IntentEngine orchestrator
├── hybrid_engine.py         # HybridEngine
├── local_engine.py          # LocalEngine
├── models/
│   ├── __init__.py
│   ├── result.py            # Result dataclass (process_voice_input output)
│   ├── response.py          # Response dataclass (generate_response output)
│   ├── audio.py             # Audio wrapper (synthesize_speech output)
│   └── decision.py          # Constitutional filter Decision dataclass
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
```

**Note:** There are NO `iml/`, `prosody/`, `emotions/`, or `accessibility/` directories. All IML parsing, prosody analysis, emotion classification, and accessibility profile handling is provided by the `prosody_protocol` package. Intent Engine imports these directly:

```python
from prosody_protocol import (
    IMLParser, IMLValidator, IMLAssembler,
    IMLDocument, Utterance, Prosody, Pause, Emphasis, Segment,
    ProsodyAnalyzer, SpanFeatures, WordAlignment, PauseInterval,
    EmotionClassifier, RuleBasedEmotionClassifier,
    IMLToSSML, AudioToIML, TextToIML, IMLToAudio,
    ProfileLoader, ProfileApplier, ProsodyProfile,
    DatasetLoader, DatasetEntry,
    Benchmark, BenchmarkReport,
)
```

### 0.2 Build System

```
pyproject.toml               # PEP 621 metadata, dependencies, entry points
```

Key decisions:
- Use `pyproject.toml` with `hatchling` as build backend (matching Prosody Protocol's choice).
- **Required dependency:** `prosody-protocol>=0.1.0a1`
- Define optional dependency groups: `whisper`, `deepgram`, `assemblyai`, `elevenlabs`, `coqui`, `espeak`, `dev`, `test`.
- Package name: `intent-engine`. Import name: `intent_engine`.

```toml
[project]
name = "intent-engine"
requires-python = ">=3.10"
dependencies = [
    "prosody-protocol>=0.1.0a1",  # IML, prosody analysis, emotion classification
]

[project.optional-dependencies]
whisper = ["openai-whisper>=2023", "prosody-protocol[audio]"]
deepgram = ["deepgram-sdk>=3.0"]
assemblyai = ["assemblyai>=0.20"]
elevenlabs = ["elevenlabs>=0.2"]
coqui = ["TTS>=0.20"]
espeak = ["pyttsx3>=2.90"]
claude = ["anthropic>=0.30"]
openai = ["openai>=1.0"]
local-llm = ["llama-cpp-python>=0.2"]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "pytest-asyncio>=0.21", "ruff>=0.1", "mypy>=1.5"]
all = ["intent-engine[whisper,deepgram,assemblyai,elevenlabs,coqui,espeak,claude,openai,local-llm,dev]"]
```

### 0.3 Development Tooling

| Tool | Purpose |
|------|---------|
| `pytest` | Test runner |
| `pytest-cov` | Coverage reporting |
| `pytest-asyncio` | Async test support |
| `mypy` | Static type checking |
| `ruff` | Linting and formatting (match PP: `target-version = "py310"`, `line-length = 100`) |
| `pre-commit` | Git hook management |

### 0.4 CI Pipeline

Set up GitHub Actions:
- **On every push:** lint (`ruff`), type-check (`mypy`), run unit tests (`pytest`).
- **On PR merge to main:** publish to PyPI (or test PyPI).

### 0.5 Deliverables Checklist

- [ ] `pyproject.toml` with `prosody-protocol` as required dependency and all optional groups
- [ ] Empty module files with `__init__.py` stubs
- [ ] `tests/` directory mirroring `intent_engine/` structure
- [ ] `.github/workflows/ci.yml`
- [ ] `pre-commit` config (`.pre-commit-config.yaml`)
- [ ] `[tool.ruff]` in `pyproject.toml` (match Prosody Protocol's settings)
- [ ] `[tool.mypy]` in `pyproject.toml`
- [ ] `Makefile` or `justfile` with common commands (`make test`, `make lint`, etc.)
- [ ] Verify `from prosody_protocol import IMLParser` works in a test

---

## Phase 1: Core Data Models (Intent Engine Only)

**Goal:** Define the Intent Engine-specific data structures that flow through the pipeline. IML data models come from `prosody_protocol` -- this phase defines only the wrapper types unique to Intent Engine.

**Prosody Protocol provides:** `IMLDocument`, `Utterance`, `Prosody`, `Pause`, `Emphasis`, `Segment`, `SpanFeatures`, `WordAlignment`, `PauseInterval`. Do NOT redefine these.

### 1.1 Result Dataclass

```python
# intent_engine/models/result.py
from dataclasses import dataclass
from prosody_protocol import IMLDocument, SpanFeatures

@dataclass(frozen=True)
class Result:
    """Output of process_voice_input()."""
    text: str                           # Plain text transcription
    emotion: str                        # Primary detected emotion
    confidence: float                   # Emotion confidence (0.0-1.0)
    iml: str                            # Serialized IML markup string
    iml_document: IMLDocument           # Parsed IML document (from prosody_protocol)
    suggested_tone: str                 # Recommended response tone
    prosody_features: list[SpanFeatures]  # Per-word prosodic features (from prosody_protocol)
    intent: str | None                  # Parsed user intent (None if not yet interpreted)
```

### 1.2 Response Dataclass

```python
# intent_engine/models/response.py
@dataclass(frozen=True)
class Response:
    """Output of generate_response()."""
    text: str        # Response text
    emotion: str     # Emotion to apply in TTS
```

### 1.3 Audio Dataclass

```python
# intent_engine/models/audio.py
@dataclass
class Audio:
    """Output of synthesize_speech()."""
    data: bytes                  # Raw audio bytes
    format: str = "wav"          # Audio format
    duration: float | None = None  # Duration in seconds
    url: str | None = None       # Cloud-hosted URL (if applicable)

    def save(self, path: str) -> None: ...
```

### 1.4 Decision Dataclass

```python
# intent_engine/models/decision.py
@dataclass(frozen=True)
class Decision:
    """Output of ConstitutionalFilter.evaluate()."""
    allow: bool
    requires_verification: bool = False
    verification_method: str | None = None
    denial_reason: str | None = None
```

### 1.5 Testing Strategy

- Unit tests for every dataclass (construction, field access, immutability).
- Verify `Result.iml_document` accepts `prosody_protocol.IMLDocument` instances.
- Verify `Result.prosody_features` accepts `list[prosody_protocol.SpanFeatures]`.

### 1.6 Deliverables Checklist

- [ ] `Result` dataclass using `prosody_protocol.IMLDocument` and `prosody_protocol.SpanFeatures`
- [ ] `Response` dataclass
- [ ] `Audio` dataclass with `.save()` method
- [ ] `Decision` dataclass
- [ ] Unit tests for all dataclasses
- [ ] Verify type compatibility with `prosody_protocol` types

---

## Phase 2: STT Module (Provider Adapters)

**Goal:** Build the speech-to-text layer with a provider-agnostic interface. Each adapter transcribes audio and returns word-level timestamps as `prosody_protocol.WordAlignment` objects.

**Prosody Protocol provides:** `WordAlignment` dataclass, `ProsodyAnalyzer` (for post-processing), `IMLAssembler` (for building IML from alignments + features). The STT adapters only need to produce word timestamps -- the Prosody Protocol handles everything from there.

### 2.1 Abstract Interface

```python
# intent_engine/stt/base.py
from prosody_protocol import WordAlignment

class STTProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio to text with word-level timestamps.

        Returns:
            TranscriptionResult with .text and .alignments (list[WordAlignment])
        """
        ...

@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    alignments: list[WordAlignment]  # From prosody_protocol
```

Each adapter converts its provider's native timestamp format into `prosody_protocol.WordAlignment(word=..., start_ms=..., end_ms=...)` objects.

### 2.2 Whisper Adapter

- Use OpenAI's Whisper model (via `openai-whisper` or `faster-whisper`).
- Extract word-level timestamps and convert to `WordAlignment` objects.
- Run locally -- no API key needed.

### 2.3 Deepgram Adapter

- Use Deepgram's SDK (`deepgram-sdk`).
- Extract word-level timestamps from the response and convert to `WordAlignment`.
- Requires API key configuration.

### 2.4 AssemblyAI Adapter

- Use AssemblyAI SDK.
- Extract word-level timestamps and convert to `WordAlignment`.
- Requires API key configuration.

### 2.5 Provider Configuration

```python
STT_PROVIDERS = {
    "whisper-prosody": WhisperSTT,
    "deepgram": DeepgramSTT,
    "assemblyai": AssemblyAISTT,
}
```

### 2.6 Integration with Prosody Protocol

After STT produces `TranscriptionResult`, the orchestrator (Phase 5) will:

```python
from prosody_protocol import ProsodyAnalyzer, IMLAssembler

analyzer = ProsodyAnalyzer()
assembler = IMLAssembler()

# STT adapter produces alignments
result = await stt.transcribe(audio_path)

# Prosody Protocol analyzes audio
features = analyzer.analyze(audio_path, result.alignments)
pauses = analyzer.detect_pauses(audio_path)

# Prosody Protocol assembles IML
iml_doc = assembler.assemble(result.alignments, features, pauses)
```

### 2.7 Testing Strategy

- Mock external APIs (Deepgram, AssemblyAI) for unit tests.
- Integration test with Whisper using a short audio clip.
- Verify output `WordAlignment` objects have valid `start_ms` < `end_ms`.

### 2.8 Deliverables Checklist

- [ ] `STTProvider` abstract base class
- [ ] `TranscriptionResult` dataclass using `prosody_protocol.WordAlignment`
- [ ] Whisper adapter
- [ ] Deepgram adapter
- [ ] AssemblyAI adapter
- [ ] Provider registry/factory
- [ ] Unit tests with mocked APIs
- [ ] Integration test with real Whisper model

---

## Phase 3: LLM Module (Intent Interpretation)

**Goal:** Build the layer that receives IML-annotated text and produces an intent interpretation and suggested response. The LLM receives IML strings (serialized by `prosody_protocol.IMLParser.to_iml_string()`) and interprets the prosodic markup.

### 3.1 Abstract Interface

```python
# intent_engine/llm/base.py

class LLMProvider(ABC):
    @abstractmethod
    async def interpret(
        self, iml_input: str, context: str | None = None
    ) -> InterpretationResult:
        """Interpret IML-annotated input and generate a response."""
        ...

@dataclass(frozen=True)
class InterpretationResult:
    intent: str
    response_text: str
    suggested_emotion: str
```

### 3.2 Prosody-Aware System Prompt

Define in `llm/prompts.py`. The prompt must teach the LLM the Prosody Protocol's IML tag set:

- `<utterance emotion="..." confidence="...">` -- overall emotional tone
- `<prosody pitch="..." pitch_contour="..." volume="..." rate="..." quality="...">` -- prosodic features
- `<emphasis level="strong|moderate|reduced">` -- stressed words
- `<pause duration="N"/>` -- significant timing gaps
- `<segment tempo="..." rhythm="...">` -- clause-level grouping

Include examples from the Prosody Protocol README showing how prosody maps to intent (e.g., `pitch_contour="fall-rise"` = sarcasm).

### 3.3 Claude Adapter

- Use the `anthropic` Python SDK.
- Send system prompt + IML-annotated user message.
- Parse structured JSON response.

### 3.4 OpenAI Adapter

- Use the `openai` Python SDK.
- Same system prompt strategy.

### 3.5 Local LLM Adapter

- Support loading GGUF models via `llama-cpp-python` or connecting to a local vLLM/Ollama server.

### 3.6 Deliverables Checklist

- [ ] `LLMProvider` abstract base class
- [ ] `InterpretationResult` dataclass
- [ ] Prosody-aware system prompt teaching the full IML tag set from Prosody Protocol
- [ ] Claude adapter
- [ ] OpenAI adapter
- [ ] Local LLM adapter
- [ ] Provider registry/factory
- [ ] Unit tests with mocked APIs
- [ ] Prompt version tracking

---

## Phase 4: TTS Module (Emotional Speech Synthesis)

**Goal:** Build the text-to-speech layer that takes response text plus an emotion label and produces naturally spoken audio with appropriate emotional tone. Optionally uses `prosody_protocol.IMLToSSML` to convert IML to SSML for providers that support it.

**Prosody Protocol provides:** `IMLToSSML` for converting IML documents to SSML. For TTS providers that accept SSML (like some ElevenLabs and Google TTS modes), use this converter directly.

### 4.1 Abstract Interface

```python
# intent_engine/tts/base.py

class TTSProvider(ABC):
    @abstractmethod
    async def synthesize(
        self, text: str, emotion: str, **kwargs
    ) -> AudioData:
        """Synthesize speech with emotional tone."""
        ...
```

### 4.2 ElevenLabs Adapter

- Use ElevenLabs API.
- Map emotion labels to ElevenLabs voice settings (stability, similarity boost, style).

### 4.3 Coqui Adapter

- Use Coqui TTS (open-source, runs locally).
- Map emotion labels to speaker embeddings or style tokens.

### 4.4 eSpeak Adapter

- Use eSpeak (lightweight, open-source).
- Use `prosody_protocol.IMLToSSML` to convert emotion parameters to SSML, then feed to eSpeak.

### 4.5 Emotion-to-Voice Parameter Mapping

Use the Prosody Protocol's core emotion vocabulary for the mapping table:

| Emotion (PP Core) | Pitch Shift | Rate | Volume | Style Notes |
|---------|-------------|------|--------|-------------|
| empathetic | -5% | 0.9x | -2dB | Warm, slightly slower |
| frustrated | +5% | 1.1x | +3dB | Tense, slightly faster |
| calm | 0% | 0.95x | 0dB | Even, measured |
| joyful | +10% | 1.15x | +2dB | Bright, upbeat |
| sarcastic | +8% | 0.95x | +1dB | Exaggerated pitch contour |
| angry | +5% | 1.2x | +6dB | Tense, fast, loud |
| sad | -8% | 0.8x | -4dB | Lower, slower, quiet |
| neutral | 0% | 1.0x | 0dB | Default baseline |

### 4.6 Deliverables Checklist

- [ ] `TTSProvider` abstract base class
- [ ] `AudioData` class with save/bytes/url/duration
- [ ] ElevenLabs adapter
- [ ] Coqui adapter
- [ ] eSpeak adapter (using `prosody_protocol.IMLToSSML` for SSML generation)
- [ ] Emotion-to-parameter mapping table (aligned with PP core vocabulary)
- [ ] Provider registry/factory
- [ ] Unit tests with mocked APIs

---

## Phase 5: Constitutional Filter

**Goal:** Build the safety system that evaluates user intent against prosodic features before allowing sensitive actions. Uses `prosody_protocol.SpanFeatures` for prosody data.

### 5.1 YAML Rule Parser

Parse constitutional rules from YAML:

```yaml
rules:
  rule_name:
    triggers: [...]
    required_prosody:
      emotion: [...]           # From PP core vocabulary
      pitch_variance: ...
      speaking_rate: [min, max]
    forbidden_prosody:
      emotion: [...]           # From PP core vocabulary
    verification:
      method: explicit_confirmation | two_factor
      retries: N
```

### 5.2 Prosody Evaluation

The evaluator receives `prosody_protocol.SpanFeatures` objects from the pipeline and checks them against rule conditions:

```python
from prosody_protocol import SpanFeatures

def evaluate(
    self,
    intent: str,
    prosody_features: list[SpanFeatures],
    context: dict | None = None,
) -> Decision:
    ...
```

Use `SpanFeatures.f0_mean`, `SpanFeatures.intensity_mean`, `SpanFeatures.speech_rate`, etc. for numeric checks. Use the emotion label from the `IMLDocument.utterances[].emotion` field for emotion checks.

### 5.3 Decision Logic

```
IF no rules match intent → Decision(allow=True)
IF rules match AND prosody passes all checks → Decision(allow=True)
IF rules match AND prosody fails AND verification defined → Decision(
    allow=False, requires_verification=True, verification_method=..., denial_reason=...
)
IF rules match AND prosody fails AND no verification → Decision(
    allow=False, requires_verification=False, denial_reason=...
)
```

### 5.4 Deliverables Checklist

- [ ] YAML rule schema definition (Pydantic model)
- [ ] Rule parser (`ConstitutionalFilter.from_yaml(path)`)
- [ ] Trigger matching engine
- [ ] Prosody evaluation using `prosody_protocol.SpanFeatures`
- [ ] `Decision` object construction
- [ ] `ConstitutionalFilter.evaluate(intent, prosody_features, context)` method
- [ ] Unit tests covering all decision branches
- [ ] Sample `constitutional_rules.yaml` for testing

---

## Phase 6: IntentEngine Orchestrator

**Goal:** Wire everything together. The `IntentEngine` class coordinates the full pipeline using Prosody Protocol components for IML assembly and Intent Engine adapters for STT/LLM/TTS.

### 6.1 Pipeline Flow

```
process_voice_input(audio_file):
    1. STT adapter transcribes audio → TranscriptionResult (text + WordAlignment list)
    2. prosody_protocol.ProsodyAnalyzer extracts features → list[SpanFeatures]
    3. prosody_protocol.ProsodyAnalyzer detects pauses → list[PauseInterval]
    4. prosody_protocol.IMLAssembler combines alignments + features + pauses → IMLDocument
    5. prosody_protocol.IMLParser serializes IMLDocument → IML string
    6. prosody_protocol.IMLValidator validates the IML string (assert zero errors)
    7. Return Result(text, emotion, confidence, iml, iml_document, suggested_tone, prosody_features, intent)

generate_response(iml, context, tone):
    1. LLM adapter interprets IML → InterpretationResult (intent + response + emotion)
    2. Return Response(text, emotion)

synthesize_speech(text, emotion):
    1. TTS adapter synthesizes text with emotion → AudioData
    2. Return Audio object
```

### 6.2 Constructor

```python
from prosody_protocol import ProsodyAnalyzer, IMLAssembler, IMLParser, IMLValidator

class IntentEngine:
    def __init__(
        self,
        stt_provider: str = "whisper-prosody",
        llm_provider: str = "claude",
        tts_provider: str = "elevenlabs",
        constitutional_rules: str | None = None,
        prosody_profile: str | None = None,
        **provider_kwargs
    ):
        # Intent Engine adapters
        self._stt = STT_PROVIDERS[stt_provider](**provider_kwargs)
        self._llm = LLM_PROVIDERS[llm_provider](**provider_kwargs)
        self._tts = TTS_PROVIDERS[tts_provider](**provider_kwargs)

        # Prosody Protocol components (NOT reimplemented)
        self._analyzer = ProsodyAnalyzer()
        self._assembler = IMLAssembler()
        self._parser = IMLParser()
        self._validator = IMLValidator()

        # Optional components
        self._filter = (
            ConstitutionalFilter.from_yaml(constitutional_rules)
            if constitutional_rules else None
        )

        # Optional accessibility profile (from prosody_protocol)
        if prosody_profile:
            from prosody_protocol import ProfileLoader, ProfileApplier
            self._profile = ProfileLoader().load(prosody_profile)
            self._profile_applier = ProfileApplier()
        else:
            self._profile = None
            self._profile_applier = None
```

### 6.3 Error Handling

- Wrap provider errors in `IntentEngineError` subclasses: `STTError`, `LLMError`, `TTSError`.
- Re-raise `prosody_protocol` errors (`AudioProcessingError`, `IMLParseError`, etc.) as-is -- they have clear error messages.
- If prosody analysis fails, fall back to text-only mode (IML without prosody tags) and log a warning.
- Always validate IML output with `IMLValidator` before returning.

### 6.4 Caching

- Cache STT results keyed by audio file hash.
- Cache prosody analysis results alongside STT results.
- Use an in-memory LRU cache initially.

### 6.5 Async Support

All public methods should be `async`. Provide synchronous wrappers for convenience.

### 6.6 Deliverables Checklist

- [ ] `IntentEngine` class using `prosody_protocol.ProsodyAnalyzer`, `IMLAssembler`, `IMLParser`, `IMLValidator`
- [ ] `process_voice_input()` full pipeline with IML validation
- [ ] `generate_response()` method
- [ ] `synthesize_speech()` method
- [ ] Optional `prosody_profile` parameter using `prosody_protocol.ProfileLoader`
- [ ] Error hierarchy (`IntentEngineError`, `STTError`, `LLMError`, `TTSError`)
- [ ] Graceful fallback when prosody analysis fails
- [ ] Result caching (LRU, keyed by audio hash)
- [ ] Async methods with sync wrappers
- [ ] End-to-end integration tests (mocked providers)

---

## Phase 7: Deployment Engines

**Goal:** Implement the deployment-specific engine variants that wrap `IntentEngine` with appropriate defaults and constraints.

### 7.1 HybridEngine

- Cloud STT/TTS, local LLM.
- Uses `prosody_protocol.ProsodyAnalyzer` locally for prosody extraction.
- Inherits from or delegates to `IntentEngine`.

### 7.2 LocalEngine

- Everything runs locally.
- Validates that all models are available on disk.
- No network calls.
- `prosody_protocol` components all run locally by default (parselmouth, librosa).

### 7.3 Deliverables Checklist

- [ ] `HybridEngine` with mixed cloud/local configuration
- [ ] `LocalEngine` with full local processing
- [ ] Shared interface: both expose `process_voice_input`, `generate_response`, `synthesize_speech`
- [ ] Configuration validation
- [ ] Unit tests for each engine variant

---

## Phase 8: Integrations and Platform Adapters

**Goal:** Provide example adapters for common voice platforms. These live in `examples/integrations/` and are not part of the core `intent_engine` package.

### 8.1 Twilio Integration

- FastAPI webhook handler for processing Twilio call recordings.

### 8.2 Slack/Discord Bot Helpers

- Audio attachment download and processing helpers.

### 8.3 Generic REST API Server

- Standalone FastAPI server exposing Intent Engine as REST endpoints.
- Endpoints: `POST /process` (audio upload), `POST /generate` (IML input), `POST /synthesize` (text + emotion).
- The `/process` endpoint returns IML validated by `prosody_protocol.IMLValidator`.

### 8.4 Deliverables Checklist

- [ ] Twilio voice webhook handler (`examples/integrations/twilio.py`)
- [ ] Slack bot helper (`examples/integrations/slack.py`)
- [ ] Discord bot helper (`examples/integrations/discord.py`)
- [ ] Generic REST API server (`examples/integrations/server.py`)
- [ ] Documentation for each integration

---

## Phase 9: Accessibility Features

**Goal:** Integrate atypical prosody profiles and augmentative communication into the engine using Prosody Protocol's profile system.

**Prosody Protocol provides:** `ProfileLoader`, `ProfileApplier`, `ProsodyProfile`, `ProsodyMapping`, plus the JSON schema at `schemas/prosody-profile.schema.json`. Do NOT reimplement these.

### 9.1 Profile Integration in IntentEngine

The `IntentEngine` constructor already accepts an optional `prosody_profile` path (Phase 6). This phase wires it through the pipeline:

```python
# In process_voice_input(), after emotion classification:
if self._profile and self._profile_applier:
    observed_features = self._derive_feature_labels(prosody_features)
    emotion, confidence = self._profile_applier.apply(
        profile=self._profile,
        features=observed_features,
        base_emotion=raw_emotion,
        base_confidence=raw_confidence,
    )
```

### 9.2 Feature Label Derivation

Bridge between `prosody_protocol.SpanFeatures` (numeric) and the profile system (categorical labels):

```python
def _derive_feature_labels(self, features: list[SpanFeatures]) -> dict[str, str]:
    """Convert numeric SpanFeatures into categorical labels for profile matching."""
    # Map f0_mean to "high" / "low" / "normal" relative to baseline
    # Map intensity_mean to "loud" / "quiet" / "normal"
    # Map speech_rate to "fast" / "slow" / "normal"
    # Map quality to the voice quality string directly
    ...
```

### 9.3 Augmentative Communication (Type-to-Speech)

```python
def type_to_speech(self, text: str, user_profile: str, emotion: str) -> Audio:
    """Convert typed text to emotionally appropriate speech."""
    from prosody_protocol import TextToIML, ProfileLoader

    predictor = TextToIML()
    iml = predictor.predict(text, context=emotion)
    return self.synthesize_speech(text, emotion)
```

### 9.4 Profile Management API

Expose helpers for CRUD operations on profiles:

```python
def create_profile(self, user_id: str, mappings: list[dict]) -> ProsodyProfile: ...
def load_profile(self, path: str) -> ProsodyProfile: ...
def validate_profile(self, profile: ProsodyProfile) -> ValidationResult: ...
```

All backed by `prosody_protocol.ProfileLoader` and `prosody_protocol.ProfileLoader.validate()`.

### 9.5 Deliverables Checklist

- [ ] Profile integration in `IntentEngine.process_voice_input()` via `prosody_protocol.ProfileApplier`
- [ ] Feature label derivation (SpanFeatures → categorical dict)
- [ ] `type_to_speech()` method using `prosody_protocol.TextToIML`
- [ ] Profile management API (create, load, validate)
- [ ] Unit tests with sample profiles (use PP's `prosody-profile.schema.json` for validation)
- [ ] Documentation on creating custom profiles

---

## Phase 10: Testing, Benchmarking, and Quality

**Goal:** Establish comprehensive test coverage, performance benchmarks, and quality gates. Use `prosody_protocol.Benchmark` and `prosody_protocol.BenchmarkReport` for accuracy evaluation.

### 10.1 Test Pyramid

| Level | Scope | Tools |
|-------|-------|-------|
| **Unit** | Individual classes and functions | `pytest`, mocks |
| **Integration** | Module interactions (STT → ProsodyAnalyzer → IMLAssembler) | `pytest`, test fixtures |
| **End-to-End** | Full pipeline with real or mocked providers | `pytest`, test audio files |
| **Contract** | IML output validates against `prosody_protocol.IMLValidator` | `prosody_protocol` |
| **Performance** | Latency and throughput benchmarks | `pytest-benchmark` |

### 10.2 IML Validation Gate

Every test that produces IML output should validate it:

```python
from prosody_protocol import IMLValidator

validator = IMLValidator()

def assert_valid_iml(iml_string: str) -> None:
    result = validator.validate(iml_string)
    assert result.valid, f"IML validation errors: {result.issues}"
```

### 10.3 Accuracy Benchmarking

Use Prosody Protocol's benchmark tools:

```python
from prosody_protocol import Benchmark, DatasetLoader

dataset = DatasetLoader().load("path/to/test_dataset.json")
benchmark = Benchmark(dataset=dataset)
report = benchmark.run(classifier=engine_classifier)
print(report.emotion_accuracy)  # Target: 87%
```

### 10.4 Performance Targets (from spec Section 9)

| Metric | Target |
|--------|--------|
| Emotion detection accuracy | 87% on prosody-protocol test set |
| Sarcasm detection accuracy | 82% on SARC dataset |
| Urgency classification accuracy | 91% on custom healthcare dataset |
| Constitutional intent verification | 96% on safety-critical scenarios |
| End-to-end latency (cloud) | < 1.2s |
| End-to-end latency (hybrid) | < 650ms |
| End-to-end latency (local GPU) | < 800ms |

### 10.5 Quality Gates for CI

- All unit tests pass.
- Type checking passes (`mypy --strict`).
- Linting passes (`ruff check`).
- All IML output validates via `prosody_protocol.IMLValidator`.
- Test coverage >= 80% (aim for 90%+ on core modules).
- No security vulnerabilities (`pip-audit`).

### 10.6 Deliverables Checklist

- [ ] Unit tests for every module (target: 90%+ coverage)
- [ ] Integration test suite
- [ ] End-to-end test with mocked providers
- [ ] IML validation gate in all tests that produce IML
- [ ] Accuracy benchmarks using `prosody_protocol.Benchmark`
- [ ] Performance benchmark suite
- [ ] CI quality gates configured
- [ ] `pip-audit` in CI

---

## Phase 11: Fine-Tuning Pipeline (Advanced)

**Goal:** Provide tooling for fine-tuning local LLMs on the Prosody Protocol dataset for improved intent interpretation.

**Prosody Protocol provides:** `DatasetLoader`, `DatasetEntry`, `Dataset` for loading training data, `MavisBridge` for converting Mavis game data, and training infrastructure in `prosody_protocol.training`. Use these as the data pipeline.

### 11.1 Scope

Optional for initial release. Enables training models that natively understand IML.

### 11.2 Components

```python
from prosody_protocol import DatasetLoader, MavisBridge

# Load training data via Prosody Protocol
loader = DatasetLoader()
dataset = loader.load("path/to/dataset.json")

# Or convert from Mavis game data
bridge = MavisBridge()
entries = bridge.convert(phoneme_events)
```

- **FineTuner class**: Wraps Hugging Face / llama.cpp fine-tuning with IML-aware data formatting.
- **Evaluation**: Use `prosody_protocol.Benchmark` to compare fine-tuned model vs. prompt-based baseline.

### 11.3 Deliverables Checklist

- [ ] Dataset loading via `prosody_protocol.DatasetLoader`
- [ ] Mavis data conversion via `prosody_protocol.MavisBridge`
- [ ] `FineTuner` class with configurable hyperparameters
- [ ] Training script (`intent_engine.training`)
- [ ] Evaluation using `prosody_protocol.Benchmark`
- [ ] Documentation on running fine-tuning

---

## Phase Dependency Graph

```
Phase 0: Scaffolding (+ prosody-protocol dependency)
    │
    ▼
Phase 1: Data Models (Intent Engine only)
    │
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
Phase 2:       Phase 3:       Phase 4:
STT Module     LLM Module     TTS Module
(produces       (consumes      (uses PP's
WordAlignment)  IML strings)   IMLToSSML)
    │              │              │
    └──────┬───────┘              │
           ▼                      │
       Phase 5:                   │
       Constitutional             │
       Filter                     │
       (uses PP's SpanFeatures)   │
           │                      │
           └──────┬───────────────┘
                  ▼
              Phase 6:
              IntentEngine Orchestrator
              (uses PP's ProsodyAnalyzer,
               IMLAssembler, IMLValidator)
                  │
        ┌─────────┼──────────┐
        ▼         ▼          ▼
    Phase 7:  Phase 8:   Phase 9:
    Deployment Integrations Accessibility
    Engines                (uses PP's
                           ProfileApplier)
        │         │          │
        └─────────┼──────────┘
                  ▼
              Phase 10:
              Testing & Benchmarking
              (uses PP's Benchmark)
                  │
                  ▼
              Phase 11:
              Fine-Tuning
              (uses PP's DatasetLoader)
```

**Key parallelism opportunities:**
- Phases 2, 3, and 4 can be developed in parallel after Phase 1 is complete.
- Phases 7, 8, and 9 can be developed in parallel after Phase 6 is complete.

---

## Recommended Execution Order

| Order | Phase | Rationale |
|-------|-------|-----------|
| 1 | Phase 0: Scaffolding | Foundation + `prosody-protocol` dependency |
| 2 | Phase 1: Data Models | Intent Engine wrapper types |
| 3 | Phase 2: STT Module | First pipeline layer (produces `WordAlignment`) |
| 4 | Phase 3: LLM Module | Second pipeline layer (consumes IML) |
| 5 | Phase 4: TTS Module | Third pipeline layer (uses `IMLToSSML`) |
| 6 | Phase 5: Constitutional Filter | Safety layer using `SpanFeatures` |
| 7 | Phase 6: Orchestrator | Wires everything with PP's `ProsodyAnalyzer` + `IMLAssembler` |
| 8 | Phase 7: Deployment Engines | Hybrid/Local packaging |
| 9 | Phase 8: Integrations | Platform-specific adapters |
| 10 | Phase 9: Accessibility | Profiles via PP's `ProfileApplier` |
| 11 | Phase 10: Testing & Benchmarks | Quality pass with PP's `Benchmark` |
| 12 | Phase 11: Fine-Tuning | Advanced, uses PP's `DatasetLoader` |

**Milestone checkpoints:**
- After Phase 2: Demo "audio in → IML out" (using PP's `ProsodyAnalyzer` + `IMLAssembler`).
- After Phase 6: Demo full pipeline "audio in → spoken response out."
- After Phase 7: Both deployment modes (Hybrid, Local) functional.
- After Phase 10: Release candidate.

---

## Prosody Protocol Compatibility Reference

Quick reference for which `prosody_protocol` components are used in each phase:

| Phase | Prosody Protocol Components Used |
|-------|------|
| 0 | `prosody-protocol` as `pyproject.toml` dependency |
| 1 | `IMLDocument`, `SpanFeatures` (type references in dataclasses) |
| 2 | `WordAlignment` (STT output format) |
| 3 | IML tag set knowledge in system prompts |
| 4 | `IMLToSSML` (for SSML-based TTS providers) |
| 5 | `SpanFeatures` (prosody evaluation in constitutional rules) |
| 6 | `ProsodyAnalyzer`, `IMLAssembler`, `IMLParser`, `IMLValidator`, `ProfileLoader`, `ProfileApplier` |
| 7 | All Phase 6 components via delegation |
| 8 | `IMLValidator` (validate API responses) |
| 9 | `ProfileLoader`, `ProfileApplier`, `ProsodyProfile`, `TextToIML` |
| 10 | `IMLValidator` (test gates), `Benchmark`, `BenchmarkReport` |
| 11 | `DatasetLoader`, `DatasetEntry`, `MavisBridge` |
