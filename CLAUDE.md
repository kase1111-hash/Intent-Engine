# CLAUDE.md

## Project Overview

Intent Engine is a prosody-aware AI system that preserves and interprets emotional intent in voice conversations. It processes audio through three layers: Speech-to-Text with prosody extraction, LLM-based intent interpretation, and emotionally-aware Text-to-Speech synthesis.

**Status:** Beta (v0.8.0) - specification/design phase. No implementation code exists yet; the project is defined by its README and spec.

**Language:** Python
**Package name:** `intent_engine`

## Repository Structure

```
/
├── README.md        # Comprehensive project documentation and vision
├── spec.md          # Technical specification
└── CLAUDE.md        # This file
```

This repo currently contains only documentation. Implementation has not yet been committed.

## Key Concepts

- **IML (Intent Markup Language):** XML-based markup that carries prosodic information (pitch, emphasis, emotion) through the pipeline
- **Prosody:** Tone, pitch, rhythm, emphasis, and other non-verbal vocal cues
- **Constitutional Filter:** Safety system that verifies genuine user intent using prosodic features before executing sensitive actions
- **Three-Layer Architecture:** STT (speech-to-text + prosody) → Intent Interpretation (LLM) → TTS (emotional speech synthesis)

## Intended Module Structure

When implementation begins, the package should follow this structure:

- `intent_engine/` - Main package
  - `IntentEngine` - Main orchestrator class
  - `CloudEngine`, `HybridEngine`, `LocalEngine` - Deployment-specific engines
  - `ConstitutionalFilter` - Rule-based intent verification
  - STT module - Speech recognition with prosody extraction (Whisper, Deepgram, AssemblyAI)
  - LLM module - Intent interpretation (Claude, OpenAI, local LLMs)
  - TTS module - Emotional speech synthesis (ElevenLabs, Coqui, eSpeak)
  - Prosody Analyzer - Pitch, energy, tempo extraction and emotion classification

## Build and Development

No build system is configured yet. When implementation starts:

- Install: `pip install intent-engine` (planned)
- The project targets Python with provider-agnostic adapters for STT, LLM, and TTS services

## Common Tasks

- Review the README for full project vision, use cases, and API examples
- Review spec.md for technical specification and API contracts
- The public API is defined in the README code examples (Section: Quick Start, Architecture Deep Dive)
- Constitutional rules are defined in YAML format (see spec.md Section 4.4)

## Code Style and Conventions

When implementing:

- Provider-agnostic design: all STT/LLM/TTS providers are interchangeable via adapters
- Emotional data is treated as sensitive PII
- Constitutional filters must evaluate prosodic features before allowing sensitive actions
- Support atypical prosody profiles for accessibility
- IML markup uses XML-like syntax with `<utterance>`, `<prosody>`, and `<emphasis>` elements
