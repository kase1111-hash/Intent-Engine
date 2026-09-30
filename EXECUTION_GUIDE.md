# Intent Engine - Execution Guide

> **Status: historical plan, reviewed 2026-09-29.** This is the phase-by-phase plan the package was built from. Phases 0-9 were implemented, Phase 10 (testing and quality) in part, and Phase 11 (fine-tuning) was cut. The code has moved on since and is the source of truth; this guide is **not kept in sync with it**. For the current API read the docstrings and `intent_engine/__init__.py`, for setup read [CONTRIBUTING.md](CONTRIBUTING.md), and for the rules an editor must keep read [CLAUDE.md](CLAUDE.md).
>
> On 2026-09-29 the statements below that were provably out of date were corrected in place, and the checklists were checked against the code: `[x]` means the item exists, `[ ]` means it was not done or was removed (with a note). The code sketches are design sketches, not API documentation; where they differ from the code, the code wins. The main differences from the original plan:
>
> - Prosody Protocol is version 0.1.0a3 and is not on PyPI; it is installed from GitHub.
> - Emotion abstains: `Result.emotion` and `Result.confidence` come from the assembled IML, and `("neutral", 0.0)` means no emotion was reported.
> - Accessibility profiles are applied by `IMLAssembler(profile=...)`, not by a feature-label bridge in the engine.
> - The constitutional filter fails closed, and the most restrictive matching rule wins.
> - The eSpeak adapter uses `pyttsx3` and plain text, not `IMLToSSML`.
> - The integrations live in `examples/integrations/` under different file names.
> - Phase 11 (fine-tuning) was removed.

A phase-by-phase plan for implementing the Intent Engine from spec to working software.

Each phase is designed to be self-contained and testable before moving to the next. Dependencies flow downward: later phases build on earlier ones.

**Critical dependency:** Intent Engine is built on the **[Prosody Protocol](https://github.com/kase1111-hash/Prosody-Protocol)** SDK (`prosody-protocol`, installed from GitHub because it is not on PyPI, `prosody_protocol` for import). The Prosody Protocol provides IML parsing, validation, data models, prosody analysis, emotion classification, accessibility profiles, and dataset tooling. Intent Engine does NOT reimplement any of these -- it provides the orchestration layer, provider adapters, constitutional filter, and deployment engines on top.

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
├── _deployment.py           # Wiring shared by HybridEngine and LocalEngine
├── errors.py                # Error hierarchy
├── py.typed                 # PEP 561 marker
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

**Note:** There are NO `iml/`, `prosody/`, `emotions/`, or `accessibility/` directories. All IML parsing, prosody analysis, emotion classification, and accessibility profile handling is provided by the `prosody_protocol` package. Intent Engine imports what it needs directly from it; the names available include:

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
- **Required dependency:** `prosody-protocol[audio]>=0.1.0a3`. It is not published to PyPI, so it is installed from GitHub before this package (see [CONTRIBUTING.md](CONTRIBUTING.md)).
- Optional dependency groups (extras): `whisper`, `deepgram`, `assemblyai`, `claude`, `openai`, `local-llm`, `elevenlabs`, `coqui`, `espeak`, `examples`, `dev` and `all`.
- Package name: `intent-engine`. Import name: `intent_engine`.

The dependency and extras section of `pyproject.toml` (which is authoritative):

```toml
[project]
name = "intent-engine"
requires-python = ">=3.10"
dependencies = [
    "prosody-protocol[audio]>=0.1.0a3",  # IML, prosody analysis, emotion classification
    "pyyaml>=6.0",
]

[project.optional-dependencies]
whisper = ["openai-whisper>=20230918", "prosody-protocol[audio]>=0.1.0a3"]
deepgram = ["deepgram-sdk>=5,<8"]
assemblyai = ["assemblyai>=0.20"]
claude = ["anthropic>=0.40"]
openai = ["openai>=1.56"]
local-llm = ["llama-cpp-python>=0.2"]
elevenlabs = ["elevenlabs>=1.8.1"]
coqui = ["coqui-tts>=0.27"]
espeak = ["pyttsx3>=2.99"]
examples = [
    "fastapi>=0.95",
    "uvicorn>=0.23",
    "python-multipart>=0.0.9",
    "httpx>=0.25",
    "prosody-protocol[api]>=0.1.0a3",
    "twilio>=8",
    "slack_sdk>=3.27",
    "discord.py>=2.3",
]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-asyncio>=0.21",
    "ruff>=0.1",
    "mypy>=1.5",
    "types-PyYAML>=6.0",
    "pre-commit>=3.0",
]
all = [
    "intent-engine[whisper,deepgram,assemblyai,claude,openai,local-llm,elevenlabs,coqui,espeak,examples,dev]",
]
```

### 0.3 Development Tooling

| Tool | Purpose |
|------|---------|
| `pytest` | Test runner |
| `pytest-cov` | Coverage reporting |
| `pytest-asyncio` | Async test support |
| `mypy` | Static type checking |
| `ruff` | Linting (match PP: `target-version = "py310"`, `line-length = 100`). the tree is not `ruff format` clean, see [CONTRIBUTING.md](CONTRIBUTING.md) |
| `pre-commit` | Git hook management (`.pre-commit-config.yaml` exists but its hooks do not work yet, see [CONTRIBUTING.md](CONTRIBUTING.md)) |

### 0.4 CI Pipeline

Set up GitHub Actions (`.github/workflows/ci.yml`, on pushes and pull requests to `main`):
- **lint job:** `ruff check` and `mypy`.
- **test job:** `pytest` with coverage (fails below 80%) on Python 3.10, 3.11 and 3.12.
- **sdk-contracts job:** the tests that need the real provider SDKs.
- **security job:** `pip-audit`.

The plan also called for publishing to PyPI when a PR is merged to `main`. CI does not publish anything, and neither package is on PyPI.

### 0.5 Deliverables Checklist

- [x] `pyproject.toml` with `prosody-protocol` as required dependency and all optional groups
- [x] Empty module files with `__init__.py` stubs (all since filled in)
- [x] `tests/` directory mirroring `intent_engine/` structure
- [x] `.github/workflows/ci.yml`
- [x] `pre-commit` config (`.pre-commit-config.yaml`; the file exists, but its hooks do not work yet)
- [x] `[tool.ruff]` in `pyproject.toml` (match Prosody Protocol's settings)
- [x] `[tool.mypy]` in `pyproject.toml`
- [x] `Makefile` or `justfile` with common commands (`make test`, `make lint`, etc.)
- [x] Verify `from prosody_protocol import IMLParser` works in a test (`tests/test_prosody_protocol_import.py`)

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
    emotion: str                        # Emotion reported in the IML; "neutral" with confidence 0.0 means none was reported
    confidence: float                   # Emotion confidence (0.0-1.0); 0.0 when none was reported
    iml: str                            # Serialized IML markup string
    iml_document: IMLDocument           # Parsed IML document (from prosody_protocol)
    suggested_tone: str                 # Tone of the user's voice worth acting on (emotion if confidence >= 0.5, else "neutral"); not the reply tone
    prosody_features: list[SpanFeatures]  # Per-span prosodic features (from prosody_protocol)
    intent: str | None = None           # None from process_voice_input(): the LLM parses the intent later (Response.intent)
```

### 1.2 Response Dataclass

```python
# intent_engine/models/response.py
@dataclass(frozen=True)
class Response:
    """Output of generate_response()."""
    text: str                  # Response text
    emotion: str               # Emotion to speak the reply with (chosen by the LLM)
    intent: str | None = None  # Intent the LLM parsed from the user's input
```

### 1.3 Audio Dataclass

```python
# intent_engine/models/audio.py
@dataclass
class Audio:
    """Output of synthesize_speech()."""
    data: bytes                  # Raw audio bytes
    format: str = "wav"          # Audio format
    sample_rate: int = 16000     # Sample rate in Hz
    duration: float | None = None  # Duration in seconds
    url: str | None = None       # Hosted URL, if any (the engine never sets it)

    def save(self, path: str | Path) -> None: ...
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

- [x] `Result` dataclass using `prosody_protocol.IMLDocument` and `prosody_protocol.SpanFeatures`
- [x] `Response` dataclass
- [x] `Audio` dataclass with `.save()` method
- [x] `Decision` dataclass
- [x] Unit tests for all dataclasses
- [x] Verify type compatibility with `prosody_protocol` types

---

## Phase 2: STT Module (Provider Adapters)

**Goal:** Build the speech-to-text layer with a provider-agnostic interface. Each adapter transcribes audio and returns word-level timestamps as `prosody_protocol.WordAlignment` objects.

**Prosody Protocol provides:** `WordAlignment` dataclass, `ProsodyAnalyzer` (for post-processing), `IMLAssembler` (for building IML from alignments + features). The STT adapters only need to produce word timestamps -- the Prosody Protocol handles everything from there.

### 2.1 Abstract Interface

```python
# intent_engine/stt/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

from prosody_protocol import WordAlignment

@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    alignments: list[WordAlignment]  # From prosody_protocol
    language: str | None = None      # Detected language code, if known

class STTProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio to text with word-level timestamps.

        Returns:
            TranscriptionResult with .text and .alignments (list[WordAlignment])
        """
        ...
```

Each adapter converts its provider's native timestamp format into `prosody_protocol.WordAlignment(word=..., start_ms=..., end_ms=...)` objects (plus `speaker` when the provider labels speakers), using the helpers in `prosody_protocol.alignment` (`from_whisper`, `from_deepgram`, `from_assemblyai`) rather than parsing timestamps itself. `transcribe()` runs on the caller's event loop, so an adapter whose SDK call blocks must run it in a worker thread (for example with `asyncio.to_thread`).

### 2.2 Whisper Adapter

- Use OpenAI's Whisper model via `openai-whisper` (the `whisper` extra). `faster-whisper` is not supported by the adapter. `openai-whisper` runs the `ffmpeg` command-line tool to load audio, so ffmpeg must be on `PATH`.
- Extract word-level timestamps and convert to `WordAlignment` objects.
- Run locally -- no API key needed.

### 2.3 Deepgram Adapter

- Use Deepgram's SDK (`deepgram-sdk>=5,<8`, through its `AsyncDeepgramClient`).
- Extract word-level timestamps from the response and convert to `WordAlignment`.
- Requires API key configuration.

### 2.4 AssemblyAI Adapter

- Use AssemblyAI SDK.
- Extract word-level timestamps and convert to `WordAlignment`.
- Requires API key configuration.

### 2.5 Provider Configuration

```python
from intent_engine.stt import STT_PROVIDERS, create_stt_provider

# STT_PROVIDERS maps each name to a dotted class path. The classes are imported
# lazily, so a missing SDK only matters for the adapter that needs it:
#   "whisper-prosody" -> "intent_engine.stt.whisper.WhisperSTT"
#   "deepgram"        -> "intent_engine.stt.deepgram.DeepgramSTT"
#   "assemblyai"      -> "intent_engine.stt.assemblyai.AssemblyAISTT"
stt = create_stt_provider("whisper-prosody", model_size="base")
```

### 2.6 Integration with Prosody Protocol

After STT produces `TranscriptionResult`, the orchestrator (Phase 6) does, in outline (`IntentEngine.process_voice_input()` runs the analysis in a worker thread and falls back to text-only IML if it fails):

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
iml_doc = assembler.assemble(result.alignments, features, pauses, language=result.language)
```

### 2.7 Testing Strategy

- Mock external APIs (Deepgram, AssemblyAI) for unit tests.
- Integration test with Whisper using a short audio clip.
- Verify output `WordAlignment` objects have valid `start_ms` < `end_ms`.

### 2.8 Deliverables Checklist

- [x] `STTProvider` abstract base class
- [x] `TranscriptionResult` dataclass using `prosody_protocol.WordAlignment`
- [x] Whisper adapter
- [x] Deepgram adapter
- [x] AssemblyAI adapter
- [x] Provider registry/factory
- [x] Unit tests with mocked APIs (and tests against the real Deepgram and AssemblyAI SDKs, which skip when they are not installed)
- [ ] Integration test with real Whisper model (not done: the Whisper tests replace the `whisper` module with a mock)

---

## Phase 3: LLM Module (Intent Interpretation)

**Goal:** Build the layer that receives IML-annotated text and produces an intent interpretation and suggested response. The LLM receives IML strings (serialized by `prosody_protocol.IMLParser.to_iml_string()`) and interprets the prosodic markup.

### 3.1 Abstract Interface

```python
# intent_engine/llm/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class InterpretationResult:
    intent: str
    response_text: str
    suggested_emotion: str

class LLMProvider(ABC):
    @abstractmethod
    async def interpret(
        self, iml_input: str, context: str | None = None
    ) -> InterpretationResult:
        """Interpret IML-annotated input and generate a response."""
        ...
```

### 3.2 Prosody-Aware System Prompt

Define in `llm/prompts.py`. The prompt must teach the LLM the Prosody Protocol's IML tag set:

- `<utterance emotion="..." confidence="...">` -- overall emotional tone
- `<prosody pitch="..." pitch_contour="..." volume="..." rate="..." quality="...">` -- prosodic features
- `<emphasis level="strong|moderate|reduced">` -- stressed words
- `<pause duration="N"/>` -- significant timing gaps
- `<segment tempo="..." rhythm="...">` -- clause-level grouping

Include examples from the Prosody Protocol README showing how prosody maps to intent (e.g., `pitch_contour="fall-rise"` = sarcasm). The prompt has a version (`PROMPT_VERSION` in `llm/prompts.py`), and every IML example in it is validated with `prosody_protocol.IMLValidator` by `tests/llm/test_prompt_iml_conformance.py`.

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

- [x] `LLMProvider` abstract base class
- [x] `InterpretationResult` dataclass
- [x] Prosody-aware system prompt teaching the full IML tag set from Prosody Protocol
- [x] Claude adapter
- [x] OpenAI adapter
- [x] Local LLM adapter
- [x] Provider registry/factory
- [x] Unit tests with mocked APIs (and tests against the real Claude SDK and a fake server, which skip when the SDK is not installed)
- [x] Prompt version tracking

---

## Phase 4: TTS Module (Emotional Speech Synthesis)

**Goal:** Build the text-to-speech layer that takes response text plus an emotion label and produces naturally spoken audio with appropriate emotional tone. The engine only sends SSML (built with `prosody_protocol.TextToIML` and `IMLToSSML`, in `type_to_speech`) to a provider that sets `supports_ssml = True`. None of the built-in adapters does, so they all receive plain text.

**Prosody Protocol provides:** `IMLToSSML` for converting IML documents to SSML. For a TTS provider that accepts SSML, use this converter directly and set `supports_ssml = True` on its adapter.

### 4.1 Abstract Interface

```python
# intent_engine/tts/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class SynthesisResult:
    audio_data: bytes
    format: str = "wav"
    sample_rate: int = 22050
    duration: float | None = None

class TTSProvider(ABC):
    supports_ssml: bool = False  # True only if synthesize() interprets SSML passed as text

    @abstractmethod
    async def synthesize(
        self, text: str, emotion: str = "neutral", **kwargs: object
    ) -> SynthesisResult:
        """Synthesize speech with emotional tone."""
        ...
```

Adapters return a `SynthesisResult`; the engine wraps it in the `Audio` model from Phase 1. An emotion outside the core vocabulary is spoken as `neutral` (`normalize_emotion`).

### 4.2 ElevenLabs Adapter

- Use ElevenLabs API.
- Map emotion labels to ElevenLabs voice settings (stability, similarity boost, style; `ELEVENLABS_EMOTION_SETTINGS` in `tts/elevenlabs.py`).

### 4.3 Coqui Adapter

- Use Coqui TTS (open-source, runs locally) through the `coqui-tts` package; the older `TTS` package cannot install on Python 3.12+.
- The emotion label only sets the `speed` passed to the model (from the rate in the table below). Whether that changes the voice depends on the model and library version. There are no speaker embeddings or style tokens.

### 4.4 eSpeak Adapter

- Use eSpeak (lightweight, open-source) through `pyttsx3` (the `espeak` extra; on Linux it needs the system eSpeak NG library).
- The emotion label adjusts the speaking rate and volume. `pyttsx3` cannot pass SSML to eSpeak, so `IMLToSSML` is not used: an SSML document passed as text is reduced to the plain text it speaks.

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
| sincere | -2% | 0.95x | 0dB | Warm, genuine |
| uncertain | +3% | 0.9x | -1dB | Rising intonation, hesitant |
| fearful | +6% | 1.15x | -2dB | Higher pitch, fast, quiet |
| surprised | +12% | 1.1x | +2dB | Sharp rise, wide pitch range |
| disgusted | -3% | 0.9x | +1dB | Low, creaky, slow |

These values are `EMOTION_VOICE_MAP` in `intent_engine/tts/base.py`. They are hand-set design values: nothing in the repository measures or tunes them against listeners. Each adapter uses only what its engine supports: ElevenLabs its own settings table, Coqui only the rate, eSpeak the rate and the volume. No built-in adapter applies the pitch shift.

### 4.6 Deliverables Checklist

- [x] `TTSProvider` abstract base class
- [x] `AudioData` class with save/bytes/url/duration (as `SynthesisResult` for adapters and `intent_engine.models.Audio` for callers)
- [x] ElevenLabs adapter
- [x] Coqui adapter
- [x] eSpeak adapter (uses `pyttsx3` and plain text, not `prosody_protocol.IMLToSSML`)
- [x] Emotion-to-parameter mapping table (aligned with PP core vocabulary)
- [x] Provider registry/factory
- [x] Unit tests with mocked APIs (and a fake-server test for ElevenLabs and real-eSpeak tests, which skip when their dependencies are missing)

---

## Phase 5: Constitutional Filter

**Goal:** Build the safety system that evaluates user intent against prosodic features before allowing sensitive actions. Uses `prosody_protocol.SpanFeatures` for prosody data.

### 5.1 YAML Rule Parser

Parse constitutional rules from YAML:

```yaml
rules:
  rule_name:
    triggers: ["delete all", "erase"]    # matched as whole word sequences against the intent
    required_prosody:                    # optional: every listed condition must hold
      emotion: ["calm", "neutral"]       # From PP core vocabulary
      pitch_variance: low                # low | normal | high (semitones of pitch movement)
      speaking_rate: [2.0, 5.0]          # [min, max] in syllables per second
    forbidden_prosody:                   # optional: blocks the action outright
      emotion: ["angry"]                 # only `emotion` is supported here
    verification:                        # optional: what to do when required_prosody fails
      method: explicit_confirmation      # or two_factor; leave `verification` out to deny outright
      retries: 2                         # accepted, but the filter does not enforce it
```

The schema is strict: an unknown key raises `ValueError` when the rules are loaded, so a condition can never be dropped silently. The rules are dataclasses in `constitutional/rules.py`, which documents the units.

### 5.2 Prosody Evaluation

The evaluator receives `prosody_protocol.SpanFeatures` objects from the pipeline and checks them against rule conditions:

```python
from prosody_protocol import SpanFeatures

def evaluate(
    self,
    intent: str,
    prosody_features: list[SpanFeatures],
    emotion: str | None = None,
    context: dict[str, object] | None = None,
    *,
    emotion_confidence: float | None = None,
    min_emotion_confidence: float = 0.5,
) -> Decision:
    ...
```

The numeric checks use `SpanFeatures.f0_contour` and `f0_range` (pitch movement) and `SpanFeatures.speech_rate`. The emotion is passed in rather than read from the IML: use `Result.emotion` and `Result.confidence`, which is what `IntentEngine.evaluate_result(intent, result)` does. `context` is accepted but not used by any rule yet.

### 5.3 Decision Logic

```
IF no rules match intent → Decision(allow=True)
FOR EACH rule that matches the intent:
    IF forbidden_prosody matches → Decision(allow=False, requires_verification=False, denial_reason=...)
    ELSE IF required_prosody fails AND verification defined → Decision(
        allow=False, requires_verification=True, verification_method=..., denial_reason=...
    )
    ELSE IF required_prosody fails → Decision(allow=False, requires_verification=False, denial_reason=...)
    ELSE → Decision(allow=True)
The most restrictive decision wins, whatever the order of the rules:
hard deny > two_factor > explicit_confirmation > allow
```

The filter fails closed. An unknown emotion (none reported, or a confidence below `min_emotion_confidence`) fails a required emotion list, and a required `pitch_variance` or `speaking_rate` that cannot be measured fails. An unknown emotion does not match a `forbidden_prosody` list.

### 5.4 Deliverables Checklist

- [x] YAML rule schema definition (dataclasses in `constitutional/rules.py`; pydantic is not a dependency)
- [x] Rule parser (`ConstitutionalFilter.from_yaml(path)`)
- [x] Trigger matching engine
- [x] Prosody evaluation using `prosody_protocol.SpanFeatures`
- [x] `Decision` object construction
- [x] `ConstitutionalFilter.evaluate(intent, prosody_features, emotion, context, ...)` method
- [x] Unit tests covering all decision branches
- [x] Sample rules for testing (`tests/constitutional/sample_rules.yaml`)

---

## Phase 6: IntentEngine Orchestrator

**Goal:** Wire everything together. The `IntentEngine` class coordinates the full pipeline using Prosody Protocol components for IML assembly and Intent Engine adapters for STT/LLM/TTS.

### 6.1 Pipeline Flow

```
process_voice_input(audio_path, use_cache=True):
    0. Look the audio up in the LRU cache (keyed by the audio content and the active profile)
    1. STT adapter transcribes audio → TranscriptionResult (text + WordAlignment list)
    2. prosody_protocol.ProsodyAnalyzer extracts features → list[SpanFeatures]        (worker thread)
    3. prosody_protocol.ProsodyAnalyzer detects pauses → list[PauseInterval]           (worker thread)
       If steps 2-3 fail, continue with text-only IML (no features, no emotion, not cached)
    4. prosody_protocol.IMLAssembler combines alignments + features + pauses → IMLDocument
       (classifies each utterance against the speaker's baseline, leaves out an emotion it
       is not confident about, and applies the active profile)
    5. prosody_protocol.IMLParser serializes IMLDocument → IML string
    6. prosody_protocol.IMLValidator validates the IML string (IntentEngineError on errors)
    7. Emotion and confidence are read back from the IMLDocument
    8. Return Result(text, emotion, confidence, iml, iml_document, suggested_tone, prosody_features)

generate_response(iml, context=None, tone=None):
    1. LLM adapter interprets IML → InterpretationResult (intent + response + emotion)
    2. Return Response(text, emotion, intent)

synthesize_speech(text, emotion="neutral"):
    1. TTS adapter synthesizes text with emotion → SynthesisResult
    2. Return Audio object
```

`tone` (typically `Result.suggested_tone`) is passed to the LLM as a hint about how the *user* sounds. It does not set the tone of the reply: the LLM chooses that from what the user needs and reports it as `Response.emotion`.

### 6.2 Constructor

```python
from typing import Any

class IntentEngine:
    def __init__(
        self,
        stt_provider: str = "whisper-prosody",
        llm_provider: str = "claude",
        tts_provider: str = "elevenlabs",
        constitutional_rules: str | None = None,   # path to a YAML rules file
        prosody_profile: str | None = None,        # path to a profile JSON, validated on load
        cache_size: int = 128,                     # cached results; 0 or less disables the cache
        stt_kwargs: dict[str, Any] | None = None,  # forwarded to the STT adapter
        llm_kwargs: dict[str, Any] | None = None,  # forwarded to the LLM adapter
        tts_kwargs: dict[str, Any] | None = None,  # forwarded to the TTS adapter
    ) -> None: ...
```

It builds the adapters with `create_stt_provider`, `create_llm_provider` and `create_tts_provider`. The Prosody Protocol components are not reimplemented: one `ProsodyAnalyzer`, `IMLParser` and `IMLValidator`, and one `RuleBasedEmotionClassifier` that the `IMLAssembler` uses (the assembler is rebuilt with `profile=` when a profile is set). `constitutional_rules` becomes `ConstitutionalFilter.from_yaml(constitutional_rules)`; an empty string is rejected rather than silently running without the filter.

### 6.3 Error Handling

- Wrap provider errors in `IntentEngineError` subclasses: `STTError`, `LLMError`, `TTSError`.
- Errors from assembling or serializing the IML (for example `ConversionError` or `IMLParseError` for STT text with control characters) are re-raised as-is -- they have clear error messages.
- If prosody analysis fails (unreadable, too short or too low-sampled audio), fall back to text-only mode (IML without prosody tags, no emotion) and log a warning. Such a result is not cached.
- Always validate IML output with `IMLValidator` before returning.

### 6.4 Caching

- Cache whole `Result`s in an in-memory LRU cache, keyed by the audio content hash and the active profile (`cache_size`, default 128; 0 or less disables it).
- Results hold transcripts and emotion, so `clear_cache()` drops them.

### 6.5 Async Support

The pipeline methods (`process_voice_input`, `generate_response`, `synthesize_speech`, `type_to_speech`) are `async`, each with a `*_sync` wrapper for convenience. The wrappers cannot be called from a running event loop (`RuntimeError`); `close()` stops their background loop early. `evaluate_intent`, `evaluate_result` and the profile methods are ordinary synchronous methods.

### 6.6 Deliverables Checklist

- [x] `IntentEngine` class using `prosody_protocol.ProsodyAnalyzer`, `IMLAssembler`, `IMLParser`, `IMLValidator`
- [x] `process_voice_input()` full pipeline with IML validation
- [x] `generate_response()` method
- [x] `synthesize_speech()` method
- [x] Optional `prosody_profile` parameter using `prosody_protocol.ProfileLoader`
- [x] Error hierarchy (`IntentEngineError`, `STTError`, `LLMError`, `TTSError`)
- [x] Graceful fallback when prosody analysis fails
- [x] Result caching (LRU, keyed by audio content and profile)
- [x] Async methods with sync wrappers
- [x] End-to-end integration tests (mocked providers)

---

## Phase 7: Deployment Engines

**Goal:** Implement the deployment-specific engine variants that wrap `IntentEngine` with appropriate defaults and constraints.

### 7.1 HybridEngine

- Cloud STT (Deepgram by default) with a local LLM (the `local` provider). TTS defaults to Coqui, which runs locally; ElevenLabs can be selected instead. `is_llm_local` reports whether the LLM really is local.
- Uses `prosody_protocol.ProsodyAnalyzer` locally for prosody extraction.
- Inherits from `IntentEngine`.

### 7.2 LocalEngine

- Defaults to local providers (Whisper, the `local` LLM, eSpeak).
- Validates that model files exist on disk, for model options that are file paths (`validate_models`).
- `is_fully_local` reports whether the configured providers keep everything on the user's machine or network. A cloud provider is accepted but logged as a warning; the engine does not block network access.
- `prosody_protocol` components run locally (Praat, through `praat-parselmouth`).

### 7.3 Deliverables Checklist

- [x] `HybridEngine` with mixed cloud/local configuration
- [x] `LocalEngine` with full local processing
- [x] Shared interface: both expose `process_voice_input`, `generate_response`, `synthesize_speech`
- [x] Configuration validation
- [x] Unit tests for each engine variant

---

## Phase 8: Integrations and Platform Adapters

**Goal:** Provide example adapters for common voice platforms. These live in `examples/integrations/` and are not part of the core `intent_engine` package.

### 8.1 Twilio Integration

- `TwilioVoiceHandler`: a framework-agnostic handler (call it from a Flask or FastAPI route) that downloads a Twilio recording, runs it through the pipeline and returns TwiML. It does not call the LLM.

### 8.2 Slack/Discord Bot Helpers

- Audio attachment download and processing helpers.

### 8.3 Generic REST API Server

- Standalone FastAPI server exposing Intent Engine as REST endpoints.
- Endpoints: `POST /process` (audio upload), `POST /generate` (IML input), `POST /synthesize` (text + emotion).
- The `/process` endpoint returns IML validated by `prosody_protocol.IMLValidator`.

### 8.4 Deliverables Checklist

- [x] Twilio voice webhook handler (`examples/integrations/twilio_voice.py`)
- [x] Slack bot helper (`examples/integrations/slack_bot.py`)
- [x] Discord bot helper (`examples/integrations/discord_bot.py`)
- [x] Generic REST API server (`examples/integrations/rest_server.py`)
- [x] Documentation for each integration (`examples/integrations/README.md`)

The modules are not named after the SDKs they wrap (`twilio.py`, `slack.py`, `discord.py`): such a file shadows the real package whenever its folder is on `sys.path`.

---

## Phase 9: Accessibility Features

**Goal:** Integrate atypical prosody profiles and augmentative communication into the engine using Prosody Protocol's profile system.

**Prosody Protocol provides:** `ProfileLoader`, `ProfileApplier`, `ProsodyProfile`, `ProsodyMapping`, plus the JSON schema at `schemas/prosody-profile.schema.json`. Do NOT reimplement these.

### 9.1 Profile Integration in IntentEngine

The `IntentEngine` constructor already accepts an optional `prosody_profile` path (Phase 6). It loads and validates the profile (`ProfileError` if it is invalid) and hands it to the assembler, which applies it to each utterance:

```python
# what IntentEngine.set_profile(profile) does, in outline
from prosody_protocol import IMLAssembler, ProfileLoader, RuleBasedEmotionClassifier

profile = ProfileLoader().load("profile.json")
assembler = IMLAssembler(emotion_classifier=RuleBasedEmotionClassifier(), profile=profile)
```

The assembler describes each utterance in the profile vocabulary, measured against the speaker's own baseline. A mapping that matches takes precedence over the classifier. The matched emotion appears in the IML, marked `x-profile="<the pattern that matched>"`, and `Result.emotion` reports it. `clear_profile()` removes the profile, and cached results are kept per profile.

### 9.2 Feature Labels

The plan bridged `SpanFeatures` (numeric) to the profile system (categorical labels) inside the engine, with a `_derive_feature_labels()` helper. That helper does not exist: the assembler describes each utterance in the profile vocabulary itself (`prosody_protocol.categorize_features`), relative to the speaker's baseline. The vocabulary is the one in `schemas/prosody-profile.schema.json`: `pitch`, `pitch_contour`, `volume`, `rate`, `quality`, `pause_frequency` and `emphasis_frequency`. Intent Engine has no feature-label code of its own.

### 9.3 Augmentative Communication (Type-to-Speech)

```python
async def type_to_speech(self, text: str, emotion: str = "neutral") -> Audio:
    """Convert typed text to emotionally appropriate speech."""
```

The text is synthesized as typed, with `emotion` shaping the voice. Only a TTS provider that reads SSML (`supports_ssml = True`) is given SSML, built with `TextToIML().predict(text, context=emotion)` and `IMLToSSML().convert(...)`; none of the built-in adapters does, so they receive the plain text. There is no `user_profile` argument: a user's profile is set with `IntentEngine(prosody_profile=...)` or `set_profile()`. `type_to_speech_sync()` is the synchronous wrapper.

### 9.4 Profile Management API

Expose helpers for CRUD operations on profiles:

```python
def create_profile(
    self, user_id: str, mappings: list[dict], description: str | None = None,
    profile_version: str = "1.0.0",
) -> ProsodyProfile: ...
def load_profile(self, path: str) -> ProsodyProfile: ...    # ProfileError if it does not validate
def validate_profile(self, profile: ProsodyProfile) -> ValidationResult: ...
def set_profile(self, profile: ProsodyProfile) -> None: ...  # applies to later process_voice_input() calls
def clear_profile(self) -> None: ...
```

All backed by `prosody_protocol.ProfileLoader` and `prosody_protocol.ProfileLoader.validate()`.

### 9.5 Deliverables Checklist

- [x] Profile integration in `IntentEngine.process_voice_input()` (through `IMLAssembler(profile=...)` rather than `prosody_protocol.ProfileApplier` directly)
- [ ] Feature label derivation (SpanFeatures → categorical dict) (not needed: the assembler does it)
- [x] `type_to_speech()` method (`prosody_protocol.TextToIML` is used only for providers that read SSML)
- [x] Profile management API (create, load, validate, and set and clear)
- [x] Unit tests with sample profiles (validated with `ProfileLoader.validate`)
- [ ] Documentation on creating custom profiles (not done)

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
| **Performance** | Latency and throughput timing of the pipeline with mocked providers (`tests/test_performance.py`); nothing times real providers or audio | `pytest` |

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

Use Prosody Protocol's benchmark tools. `Benchmark` runs a converter (an `AudioToIML`, or any object with `convert(audio_path)` that returns an IML string) over a labelled dataset and scores the result:

```python
from prosody_protocol import AudioToIML, Benchmark, DatasetLoader

dataset = DatasetLoader().load("path/to/dataset")  # a directory containing entries/*.json
report = Benchmark(dataset, converter=AudioToIML()).run()
print(report.emotion_accuracy)
```

`tests/test_benchmarks.py` exercises this harness with synthetic dataset entries and a stand-in converter that returns each entry's own annotation, so it scores 100% by construction. No accuracy of this pipeline has been measured.

### 10.4 Performance Targets (from spec Section 9; untested)

These are targets only. Nothing in this repository measures them: no benchmark has been run with real audio, and the datasets named below are not available. SARC, a custom healthcare set and safety-critical scenarios are not in this repository, and Prosody Protocol ships no test corpus yet (its own datasets folder says only small test fixtures exist).

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

- [x] Unit tests for every module (CI enforces 80% coverage; it measured 99% when last run)
- [x] Integration test suite (`tests/test_integration.py`)
- [x] End-to-end test with mocked providers (`tests/test_end_to_end.py`)
- [x] IML validation gate in all tests that produce IML (`assert_valid_iml` in `tests/conftest.py` and `tests/test_iml_contract.py`; not checked for every test)
- [ ] Accuracy benchmarks using `prosody_protocol.Benchmark` (only the harness is tested, with synthetic entries; see 10.3)
- [ ] Performance benchmark suite (`tests/test_performance.py` times the pipeline with mocked providers only)
- [x] CI quality gates configured
- [x] `pip-audit` in CI

---

## Phase 11: Fine-Tuning Pipeline (Advanced) -- removed

**Status:** cut in commit 54c12c9 ("Refocus: cut CloudEngine, training wrappers; move integrations to examples"). There is no `intent_engine.training` package, no `FineTuner` class and no training script in this repository, and none of the items in this phase exist.

**Goal (not pursued):** tooling for fine-tuning local LLMs on datasets in the Prosody Protocol format (upstream ships no corpus yet), so that models natively understand IML. Prosody Protocol has the data side: `DatasetLoader().load(dataset_dir)` takes a dataset directory (not a single JSON file), `MavisBridge` converts Mavis game data (`phoneme_events_to_entry`, `export_dataset`), and `Benchmark` scores a converter. Its training tooling is the `training/` folder of its source repository, not an importable `prosody_protocol.training` module.

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
(produces       (consumes      (SSML only if
WordAlignment)  IML strings)   supported)
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
                           IMLAssembler)
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
              (removed)
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
| 5 | Phase 4: TTS Module | Third pipeline layer (`IMLToSSML` only for providers that read SSML) |
| 6 | Phase 5: Constitutional Filter | Safety layer using `SpanFeatures` |
| 7 | Phase 6: Orchestrator | Wires everything with PP's `ProsodyAnalyzer` + `IMLAssembler` |
| 8 | Phase 7: Deployment Engines | Hybrid/Local packaging |
| 9 | Phase 8: Integrations | Platform-specific adapters |
| 10 | Phase 9: Accessibility | Profiles via PP's `IMLAssembler(profile=...)` |
| 11 | Phase 10: Testing & Benchmarks | Quality pass with PP's `Benchmark` |
| 12 | Phase 11: Fine-Tuning | Advanced; removed from the repository |

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
| 2 | `WordAlignment` (STT output format), `prosody_protocol.alignment` (`from_whisper`, `from_deepgram`, `from_assemblyai`) |
| 3 | IML tag set knowledge in system prompts |
| 4 | `IMLToSSML` (only for a provider that sets `supports_ssml`; no built-in adapter does) |
| 5 | `SpanFeatures` (prosody evaluation in constitutional rules) |
| 6 | `ProsodyAnalyzer`, `IMLAssembler`, `IMLParser`, `IMLValidator`, `RuleBasedEmotionClassifier`, `ProfileLoader` |
| 7 | All Phase 6 components via delegation |
| 8 | `IMLValidator` (validate API responses) |
| 9 | `ProfileLoader`, `ProsodyProfile`, `ProsodyMapping`, `IMLAssembler(profile=...)`, `TextToIML` |
| 10 | `IMLValidator` (test gates), `Benchmark`, `BenchmarkReport` |
| 11 | (removed) |
