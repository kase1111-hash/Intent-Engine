# CLAUDE.md

## Project Overview

Intent Engine is a prosody-aware AI system that preserves and interprets emotional intent in voice conversations. It processes audio through three layers: Speech-to-Text with prosody extraction, LLM-based intent interpretation, and emotionally-aware Text-to-Speech synthesis.

**Status:** Beta (v0.8.0) - Core pipeline, constitutional filter, and provider adapters implemented. Refocused from broad feature set to core pipeline validation.

**Language:** Python
**Package name:** `intent_engine`

## Prosody Protocol Dependency (CRITICAL)

Intent Engine is built on top of the **Prosody Protocol** SDK. This is the canonical source for IML (Intent Markup Language) parsing, validation, prosody analysis, emotion classification, and accessibility profiles.

**Repository:** https://github.com/kase1111-hash/Prosody-Protocol
**Package:** `prosody-protocol` / `prosody_protocol` (import). It is **not published to PyPI**: install it from GitHub, with the `[audio]` extra (add `[api]` for the example REST server).
**Version:** 0.1.0a3, commit `4d4f0bb930b33f5d66015f8565a87e16c02e5fe2` (the one CI pins as `PROSODY_PROTOCOL` in `.github/workflows/ci.yml`). `pyproject.toml` requires `prosody-protocol[audio]>=0.1.0a3`.
**IML version:** the IML this SDK writes carries `version="0.1.0"` on `<iml>` (a semantic version; IML specification 0.1.0-alpha). The schema files are versioned separately: `schemas/iml-1.0.xsd` is schema version 1.0.0 and describes IML specification 0.1.0-alpha, so the "1.0" in its name is not the IML document version.

### What Prosody Protocol Provides (DO NOT reimplement)

Intent Engine **MUST** use `prosody_protocol` for all of the following. Never build custom versions of these components:

| Component | Prosody Protocol Class | Purpose |
|---|---|---|
| IML Parsing | `IMLParser` | Parse IML XML strings into `IMLDocument` objects |
| IML Validation | `IMLValidator`, `ValidationResult`, `ValidationIssue` | Validate IML against spec rules V1-V33 (errors, warnings and info notes) |
| IML Data Models | `IMLDocument`, `Utterance`, `Prosody`, `Pause`, `Emphasis`, `Segment` | Immutable dataclasses for IML elements |
| IML Assembly | `IMLAssembler` | Build `IMLDocument` from STT alignments + prosody features; classifies each utterance against the speaker's baseline, leaves out emotions below its confidence threshold, and applies a prosody profile (`IMLAssembler(profile=...)`) |
| STT Word Timings | `prosody_protocol.alignment`: `from_whisper`, `from_deepgram`, `from_assemblyai`, `from_google`, `from_records` (`load_word_timings` and `parse_word_timings` are also exported from the top level) | Convert a speech recognizer's response into `WordAlignment` lists; STT adapters use these rather than parsing timestamps themselves |
| Prosody Analysis | `ProsodyAnalyzer`, `SpanFeatures`, `WordAlignment`, `PauseInterval` | Extract F0, intensity, jitter, shimmer, HNR from audio |
| Emotion Classification | `EmotionClassifier` (protocol), `RuleBasedEmotionClassifier`, `BaselineAwareEmotionClassifier`, `SpeakerBaseline` | Classify emotion from prosodic features. The rule-based classifier labels only `neutral`, `calm`, `sad`, `angry`, `joyful` and `fearful`, and without a baseline it returns `("neutral", 0.0)` |
| IML to SSML | `IMLToSSML` | Convert IML to SSML for TTS engines |
| Audio to IML | `AudioToIML` | End-to-end audio-to-IML conversion |
| Text to IML | `TextToIML` | Predict prosody for plain text |
| IML to Audio | `IMLToAudio` | Synthesize waveforms from IML |
| Accessibility Profiles | `ProfileLoader`, `ProfileApplier`, `ProsodyProfile`, `ProsodyMapping`, `ProfileMatch`, `categorize_features` | Atypical prosody profile management |
| Datasets | `DatasetLoader`, `DatasetEntry`, `Dataset` | Load and validate training datasets (upstream ships no corpora yet, only small test fixtures) |
| Benchmarking | `Benchmark`, `BenchmarkReport` | Evaluate model accuracy |
| Mavis Bridge | `MavisBridge`, `PhonemeEvent` | Convert Mavis game data to datasets |
| Exceptions | `ProsodyProtocolError`, `IMLParseError`, `IMLValidationError`, `ProfileError`, `AudioProcessingError`, `ConversionError`, `DatasetError`, `SpeechRecognitionError`, `TrainingError` | Error hierarchy |

Also available upstream, and not currently used by Intent Engine: `to_llm_context` and `build_messages` (format an IML document as an annotated transcript and chat messages for an LLM; the adapters here send IML with their own system prompt, `intent_engine/llm/prompts.py`). Check the Prosody Protocol before writing any new IML, audio or LLM-formatting helper.

### What Intent Engine Adds (our value-add)

Intent Engine provides the **orchestration layer** and **provider adapters** that the Prosody Protocol does not:

- `IntentEngine` orchestrator (wires STT + prosody + LLM + TTS together)
- `HybridEngine`, `LocalEngine` deployment modes
- `ConstitutionalFilter` safety governance (YAML rules, prosody-based intent verification)
- STT provider adapters (Whisper, Deepgram, AssemblyAI)
- LLM provider adapters (Claude, OpenAI, local LLMs) with prosody-aware prompts
- TTS provider adapters (ElevenLabs, Coqui, eSpeak) with emotion-to-voice mapping
- Example integrations in `examples/` (Twilio, Slack, Discord, REST API server)

### IML Compatibility Rules

1. All IML output MUST validate against `prosody_protocol.IMLValidator` with zero errors
2. All IML output MUST conform to the XSD schema at `schemas/iml-1.0.xsd` in the Prosody Protocol repo. The XSD cannot express every rule (for example V3, `confidence` with `emotion`), so `IMLValidator` is the authority
3. Use Prosody Protocol's core emotion vocabulary: `neutral`, `sincere`, `sarcastic`, `frustrated`, `joyful`, `uncertain`, `angry`, `sad`, `fearful`, `surprised`, `disgusted`, `calm`, `empathetic`. The built-in classifier only emits `neutral`, `calm`, `sad`, `angry`, `joyful` and `fearful`; the other labels can only come from an LLM or a prosody profile
4. Custom emotions (e.g., `confident`, `deliberate`, `rushed`) are allowed but will trigger V15 info-level validation notices
5. Prosody profiles MUST conform to `schemas/prosody-profile.schema.json`
6. Dataset entries MUST conform to `schemas/dataset-entry.schema.json`
7. IML documents use `<iml>` as root wrapper or standalone `<utterance>` elements
8. The `<segment>` element MUST only appear as a direct child of `<utterance>` (not nested)
9. `confidence` attribute is REQUIRED when `emotion` is present on `<utterance>`
10. `<pause>` elements MUST be self-closing with a positive integer `duration` in milliseconds
11. The `version` attribute of `<iml>` is a semantic version (V28) and `language` is a BCP 47 tag (V29). The pipeline writes `version="0.1.0"`; never write `1.0`
12. An utterance with no reliable emotion carries no `emotion` and no `confidence` attribute (the assembler abstains below its confidence threshold). Never add `emotion="neutral"` with an invented confidence to fill the gap

## Repository Structure

```
/
├── intent_engine/          # Main package (the only thing the wheel ships)
│   ├── __init__.py         # Public API exports
│   ├── engine.py           # IntentEngine orchestrator
│   ├── errors.py           # Error hierarchy
│   ├── hybrid_engine.py    # HybridEngine (cloud STT and local LLM by default)
│   ├── local_engine.py     # LocalEngine (local providers by default)
│   ├── _deployment.py      # Wiring shared by HybridEngine and LocalEngine (model options, is-local checks)
│   ├── py.typed            # PEP 561 marker
│   ├── models/             # Result, Response, Audio, Decision dataclasses (one module each)
│   ├── stt/                # base.py + STT adapters (whisper.py, deepgram.py, assemblyai.py)
│   ├── llm/                # base.py, prompts.py + LLM adapters (claude.py, openai.py, local.py)
│   ├── tts/                # base.py + TTS adapters (elevenlabs.py, coqui.py, espeak.py)
│   └── constitutional/     # Constitutional filter (filter.py, rules.py YAML schema, evaluator.py)
├── examples/               # Example integrations; not part of the installed package
│   ├── __init__.py
│   └── integrations/       # rest_server.py, twilio_voice.py, slack_bot.py, discord_bot.py,
│                           # _common.py, README.md, conftest.py and the test_*.py files (importable
│                           # as examples.integrations.<module> from the repo root)
├── tests/                  # pytest suite: test_*.py for the engines and pipeline at the top level,
│                           # constitutional/, llm/, models/, stt/, tts/ per subpackage, and
│                           # synth_audio.py (generates the test audio)
├── pyproject.toml          # Build config (hatchling), dependencies, extras, pytest/ruff/mypy settings
├── Makefile                # Dev shortcuts (dev, test, lint, typecheck, format, check, clean)
├── .github/workflows/ci.yml  # GitHub Actions CI pipeline
├── .pre-commit-config.yaml # Pre-commit hooks (not usable yet, see CONTRIBUTING.md)
├── .gitignore
├── README.md               # Project documentation and vision
├── spec.md                 # Technical specification
├── EXECUTION_GUIDE.md      # Phase-by-phase implementation plan (historical)
├── EVALUATION.md           # Repository evaluation (a dated snapshot)
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

## Behaviour Contracts (do not break)

- **Emotion abstains.** `Result.emotion`, `Result.confidence` and `Result.suggested_tone` come from the assembled IML. `("neutral", 0.0)` means *no emotion was reported* (the assembler abstains; a single utterance without calibration speech cannot report one), not "measured neutral". `suggested_tone` describes the **user** (`emotion` when `confidence >= 0.5`, else `"neutral"`); the tone to speak the reply in is `Response.emotion`, which the LLM chooses. `Response.intent` is the intent the LLM parsed; `Result.intent` stays `None` because `process_voice_input()` runs before the LLM.
- **The constitutional filter fails closed.** The gate is `IntentEngine.evaluate_result(response.intent, result)`: it weighs every emotion the assembler reported for the turn (a forbidden emotion in any sentence denies; every reported emotion must satisfy a required list). `evaluate_intent` sees only the one emotion passed to it and, without `emotion_confidence`, reads an abstention as a measured `neutral`, so do not gate with it. An unknown emotion (an abstention, or a confidence below `min_emotion_confidence`, default 0.5) fails a required emotion list, and a required pitch or speech-rate condition that cannot be measured fails. Every matching rule is evaluated and the most restrictive decision wins. Triggers match whole word sequences (`"delete all"` matches `delete_all_files`, not `undelete_all`). The rules schema is strict: unknown keys raise `ValueError`. An engine built without `constitutional_rules` has no filter and allows everything. Do not weaken any of this.
- **Async API.** `process_voice_input`, `generate_response`, `synthesize_speech` and `type_to_speech` are coroutines. The `*_sync` wrappers cannot be called from a running event loop (`RuntimeError`). Adapters must not block the loop.

## Module Structure

- `intent_engine/` - Main package
  - `IntentEngine` - Main orchestrator class
  - `HybridEngine`, `LocalEngine` - Deployment-specific engines (shared wiring in `_deployment.py`)
  - `ConstitutionalFilter` - Rule-based intent verification
  - `stt/` - Speech recognition adapters (Whisper, Deepgram, AssemblyAI)
  - `llm/` - Intent interpretation adapters (Claude, OpenAI, local LLMs)
  - `tts/` - Speech synthesis adapters (ElevenLabs, Coqui, eSpeak)
  - `models/` - Dataclasses: Result, Response and Decision are frozen; Audio is a plain (mutable) dataclass
  - `constitutional/` - YAML rules, evaluator, filter
  - Prosody analysis uses **`prosody_protocol.ProsodyAnalyzer`** (not reimplemented)
  - IML parsing uses **`prosody_protocol.IMLParser` / `IMLAssembler`** (not reimplemented)
  - Emotion classification uses **`prosody_protocol.RuleBasedEmotionClassifier`** (not reimplemented)
- `examples/` - Example platform integrations (Twilio, Slack, Discord, REST server)

## Build and Development

- Python >=3.10 (CI tests 3.10, 3.11 and 3.12), with provider-agnostic adapters for STT, LLM, and TTS services
- Install: `prosody-protocol` is not on PyPI (and neither is `intent-engine`), so install the pinned commit from GitHub first, then the package: `pip install "prosody-protocol[audio] @ git+https://github.com/kase1111-hash/Prosody-Protocol.git@<commit from PROSODY_PROTOCOL in .github/workflows/ci.yml>"`, then `pip install -e ".[dev]"`. `make dev` does both (`make install` does the same without the dev tools)
- Core dependency: `prosody-protocol[audio]>=0.1.0a3` (required, listed in `pyproject.toml`)
- Extras (`pyproject.toml` is authoritative): `whisper`, `deepgram` (deepgram-sdk>=5,<8), `assemblyai`, `claude`, `openai`, `local-llm`, `elevenlabs`, `coqui` (the `coqui-tts` fork; the older `TTS` 0.22 cannot install on Python 3.12+), `espeak` (`pyttsx3>=2.99` plus the system eSpeak NG library), `examples` (what the example integrations import), `dev`, and `all`
- Run tests: `make test` (pytest with coverage; fails under 80%, as CI does). Plain `pytest` collects `tests/` and `examples/`. Tests that need a provider SDK or an example's web framework skip themselves when it is not installed. To run them all, as CI's `sdk-contracts` job does, install `prosody-protocol[audio,api]` from GitHub, then `pip install -e ".[dev,claude,openai,deepgram,assemblyai,elevenlabs,examples]"`, then run `pytest` (no network or API keys needed)
- Lint: `make lint` (ruff check on `intent_engine/`, `tests/` and `examples/`)
- Type check: `make typecheck` (mypy --strict on `intent_engine/`)
- Full check: `make check` (lint + typecheck + test)
- Format: the tree is not `ruff format` clean, so never run it over the whole repo. `make format FILES="path/a.py path/b.py"` formats only the files you name
- Pre-commit: `.pre-commit-config.yaml` does not work yet (its mypy hook cannot install prosody-protocol from PyPI); do not run `pre-commit install`. See CONTRIBUTING.md

## Common Tasks

- Review the README for full project vision, use cases, and API examples
- Review spec.md for technical specification and API contracts
- EXECUTION_GUIDE.md is the original phased implementation plan, kept as history; the code is the source of truth
- Review the Prosody Protocol repo (https://github.com/kase1111-hash/Prosody-Protocol) for IML spec, schemas, and SDK API
- The public API is what `intent_engine/__init__.py` exports (`IntentEngine`, `HybridEngine`, `LocalEngine`, `Result`, `Response`, `Audio`, `Decision`, `ConstitutionalFilter` and the error classes); the README code examples (Quick Start, Architecture Deep Dive) show how to use it
- Constitutional rules are defined in YAML format (see spec.md Section 4.4; the schema is documented in `intent_engine/constitutional/rules.py`, and `tests/constitutional/sample_rules.yaml` is an example)

## Code Style and Conventions

When implementing:

- Provider-agnostic design: all STT/LLM/TTS providers are interchangeable via adapters
- Always use `prosody_protocol` types for IML data (`IMLDocument`, `Utterance`, `Prosody`, etc.) -- never define parallel types
- Always validate IML output with `prosody_protocol.IMLValidator` before returning to callers
- Emotional data is treated as sensitive PII
- Constitutional filters must evaluate prosodic features before allowing sensitive actions, and fail closed when the evidence is missing (see Behaviour Contracts)
- Support atypical prosody profiles via `prosody_protocol.ProfileLoader` (load and validate) and `IMLAssembler(profile=...)`, which applies the profile per utterance and marks it `x-profile` in the IML
- IML markup uses XML syntax with `<utterance>`, `<prosody>`, `<pause>`, `<emphasis>`, and `<segment>` elements as defined by the Prosody Protocol spec
