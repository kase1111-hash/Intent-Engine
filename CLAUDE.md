# CLAUDE.md

## Project Overview

Intent Engine is a prosody-aware AI system that preserves and interprets emotional intent in voice conversations. It processes audio through three layers: Speech-to-Text with prosody extraction, LLM-based intent interpretation, and emotionally-aware Text-to-Speech synthesis.

**Status:** Beta (v0.8.0) - All 10 implementation phases complete. 38 source files, 37 test files, 571 tests.

**Language:** Python
**Package name:** `intent_engine`

## Prosody Protocol Dependency (CRITICAL)

Intent Engine is built on top of the **Prosody Protocol** SDK. This is the canonical source for IML (Intent Markup Language) parsing, validation, prosody analysis, emotion classification, and accessibility profiles.

**Repository:** https://github.com/kase1111-hash/Prosody-Protocol
**Package:** `prosody-protocol` (PyPI) / `prosody_protocol` (import)
**Version:** 0.1.0-alpha (tracks IML spec v1.0)

### What Prosody Protocol Provides (DO NOT reimplement)

Intent Engine **MUST** use `prosody_protocol` for all of the following. Never build custom versions of these components:

| Component | Prosody Protocol Class | Purpose |
|---|---|---|
| IML Parsing | `IMLParser` | Parse IML XML strings into `IMLDocument` objects |
| IML Validation | `IMLValidator`, `ValidationResult`, `ValidationIssue` | Validate IML against spec rules V1-V18 |
| IML Data Models | `IMLDocument`, `Utterance`, `Prosody`, `Pause`, `Emphasis`, `Segment` | Immutable dataclasses for IML elements |
| IML Assembly | `IMLAssembler` | Build `IMLDocument` from STT alignments + prosody features |
| Prosody Analysis | `ProsodyAnalyzer`, `SpanFeatures`, `WordAlignment`, `PauseInterval` | Extract F0, intensity, jitter, shimmer, HNR from audio |
| Emotion Classification | `EmotionClassifier` (protocol), `RuleBasedEmotionClassifier` | Classify emotion from prosodic features |
| IML to SSML | `IMLToSSML` | Convert IML to SSML for TTS engines |
| Audio to IML | `AudioToIML` | End-to-end audio-to-IML conversion |
| Text to IML | `TextToIML` | Predict prosody for plain text |
| IML to Audio | `IMLToAudio` | Synthesize waveforms from IML |
| Accessibility Profiles | `ProfileLoader`, `ProfileApplier`, `ProsodyProfile`, `ProsodyMapping` | Atypical prosody profile management |
| Datasets | `DatasetLoader`, `DatasetEntry`, `Dataset` | Load and validate training datasets |
| Benchmarking | `Benchmark`, `BenchmarkReport` | Evaluate model accuracy |
| Mavis Bridge | `MavisBridge`, `PhonemeEvent` | Convert Mavis game data to datasets |
| Exceptions | `ProsodyProtocolError`, `IMLParseError`, `IMLValidationError`, `ProfileError`, `AudioProcessingError`, `ConversionError`, `DatasetError`, `TrainingError` | Error hierarchy |

### What Intent Engine Adds (our value-add)

Intent Engine provides the **orchestration layer** and **provider adapters** that the Prosody Protocol does not:

- `IntentEngine` orchestrator (wires STT + prosody + LLM + TTS together)
- `CloudEngine`, `HybridEngine`, `LocalEngine` deployment modes
- `ConstitutionalFilter` safety governance (YAML rules, prosody-based intent verification)
- STT provider adapters (Whisper, Deepgram, AssemblyAI)
- LLM provider adapters (Claude, OpenAI, local LLMs) with prosody-aware prompts
- TTS provider adapters (ElevenLabs, Coqui, eSpeak) with emotion-to-voice mapping
- Platform integrations (Twilio, Slack, Discord, REST API server)

### IML Compatibility Rules

1. All IML output MUST validate against `prosody_protocol.IMLValidator` with zero errors
2. All IML output MUST conform to the XSD schema at `schemas/iml-1.0.xsd` in the Prosody Protocol repo
3. Use Prosody Protocol's core emotion vocabulary: `neutral`, `sincere`, `sarcastic`, `frustrated`, `joyful`, `uncertain`, `angry`, `sad`, `fearful`, `surprised`, `disgusted`, `calm`, `empathetic`
4. Custom emotions (e.g., `confident`, `deliberate`, `rushed`) are allowed but will trigger V15 info-level validation notices
5. Prosody profiles MUST conform to `schemas/prosody-profile.schema.json`
6. Dataset entries MUST conform to `schemas/dataset-entry.schema.json`
7. IML documents use `<iml>` as root wrapper or standalone `<utterance>` elements
8. The `<segment>` element MUST only appear as a direct child of `<utterance>` (not nested)
9. `confidence` attribute is REQUIRED when `emotion` is present on `<utterance>`
10. `<pause>` elements MUST be self-closing with a positive integer `duration` in milliseconds

## Repository Structure

```
/
├── intent_engine/          # Main package (38 source files)
│   ├── __init__.py         # Public API exports
│   ├── engine.py           # IntentEngine orchestrator
│   ├── errors.py           # Error hierarchy
│   ├── cloud_engine.py     # CloudEngine (managed API client)
│   ├── hybrid_engine.py    # HybridEngine (cloud STT/TTS + local LLM)
│   ├── local_engine.py     # LocalEngine (fully local)
│   ├── models/             # Result, Response, Audio, Decision dataclasses
│   ├── stt/                # STT adapters (Whisper, Deepgram, AssemblyAI)
│   ├── llm/                # LLM adapters (Claude, OpenAI, local)
│   ├── tts/                # TTS adapters (ElevenLabs, Coqui, eSpeak)
│   ├── constitutional/     # Constitutional filter (YAML rules, evaluator)
│   └── integrations/       # Platform integrations (Twilio, Slack, Discord, REST)
├── tests/                  # 37 test files, 571 tests
├── pyproject.toml          # Build config (hatchling), dependencies, tooling
├── Makefile                # Dev shortcuts (install, test, lint, typecheck)
├── .github/workflows/ci.yml  # GitHub Actions CI pipeline
├── README.md               # Project documentation and vision
├── spec.md                 # Technical specification
├── EXECUTION_GUIDE.md      # Phase-by-phase implementation plan
├── CONTRIBUTING.md         # Contribution guidelines
├── LICENSE                 # Apache 2.0
└── CLAUDE.md               # This file
```

## Key Concepts

- **IML (Intent Markup Language):** XML-based markup defined by the Prosody Protocol that carries prosodic information (pitch, emphasis, emotion) through the pipeline
- **Prosody Protocol:** The upstream SDK (https://github.com/kase1111-hash/Prosody-Protocol) that defines IML and provides parsing, validation, analysis, and classification tools
- **Prosody:** Tone, pitch, rhythm, emphasis, and other non-verbal vocal cues
- **Constitutional Filter:** Safety system that verifies genuine user intent using prosodic features before executing sensitive actions
- **Three-Layer Architecture:** STT (speech-to-text + prosody) → Intent Interpretation (LLM) → TTS (emotional speech synthesis)

## Module Structure

- `intent_engine/` - Main package
  - `IntentEngine` - Main orchestrator class
  - `CloudEngine`, `HybridEngine`, `LocalEngine` - Deployment-specific engines
  - `ConstitutionalFilter` - Rule-based intent verification
  - `stt/` - Speech recognition adapters (Whisper, Deepgram, AssemblyAI)
  - `llm/` - Intent interpretation adapters (Claude, OpenAI, local LLMs)
  - `tts/` - Speech synthesis adapters (ElevenLabs, Coqui, eSpeak)
  - `models/` - Frozen dataclasses: Result, Response, Audio, Decision
  - `constitutional/` - YAML rules, evaluator, filter
  - `integrations/` - Twilio, Slack, Discord, REST API server
  - Prosody analysis uses **`prosody_protocol.ProsodyAnalyzer`** (not reimplemented)
  - IML parsing uses **`prosody_protocol.IMLParser` / `IMLAssembler`** (not reimplemented)
  - Emotion classification uses **`prosody_protocol.RuleBasedEmotionClassifier`** (not reimplemented)

## Build and Development

- Install: `pip install -e .` (or `pip install intent-engine` from PyPI when published)
- Dev install: `pip install -e ".[dev]"` or `make dev`
- Core dependency: `prosody-protocol>=0.1.0a1` (required, listed in `pyproject.toml`)
- Run tests: `make test` (requires 80% coverage)
- Lint: `make lint` (ruff)
- Type check: `make typecheck` (mypy --strict)
- Full check: `make check` (lint + typecheck + test)
- The project targets Python >=3.10 with provider-agnostic adapters for STT, LLM, and TTS services

## Common Tasks

- Review the README for full project vision, use cases, and API examples
- Review spec.md for technical specification and API contracts
- Review EXECUTION_GUIDE.md for the phased implementation plan
- Review the Prosody Protocol repo (https://github.com/kase1111-hash/Prosody-Protocol) for IML spec, schemas, and SDK API
- The public API is defined in the README code examples (Section: Quick Start, Architecture Deep Dive)
- Constitutional rules are defined in YAML format (see spec.md Section 4.4)

## Code Style and Conventions

When implementing:

- Provider-agnostic design: all STT/LLM/TTS providers are interchangeable via adapters
- Always use `prosody_protocol` types for IML data (`IMLDocument`, `Utterance`, `Prosody`, etc.) -- never define parallel types
- Always validate IML output with `prosody_protocol.IMLValidator` before returning to callers
- Emotional data is treated as sensitive PII
- Constitutional filters must evaluate prosodic features before allowing sensitive actions
- Support atypical prosody profiles via `prosody_protocol.ProfileLoader` and `prosody_protocol.ProfileApplier`
- IML markup uses XML syntax with `<utterance>`, `<prosody>`, `<pause>`, `<emphasis>`, and `<segment>` elements as defined by the Prosody Protocol spec
