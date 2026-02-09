## PROJECT EVALUATION REPORT

**Primary Classification:** Underdeveloped
**Secondary Tags:** Feature Creep risk, Good Concept

---

### CONCEPT ASSESSMENT

**What real problem does this solve?**

Voice AI systems discard prosodic information (pitch, rhythm, emphasis, emotion) during speech-to-text, losing 60-80% of communicative intent. A customer saying "Oh great, another meeting" with sarcastic intonation gets the same flat transcript as genuine enthusiasm. Intent Engine proposes to carry structured prosodic metadata through the entire pipeline -- STT to LLM reasoning to TTS output -- using IML (Intent Markup Language) as the transport format.

**Who is the user? Is the pain real or optional?**

Developers building voice-based AI products (customer support bots, healthcare screening, accessibility tools). The pain is real for anyone building production voice AI -- misread sarcasm, missed urgency, and tone-deaf responses are measurable failure modes. However, most voice AI products today work around this by training on massive audio datasets directly (Hume AI, ElevenLabs), not by structuring prosody into markup. The structured-metadata approach is academically compelling but commercially unproven.

**Is this solved better elsewhere?**

Not in the same way. Hume AI does emotion detection from audio. ElevenLabs does emotional TTS. Amazon and Google have sentiment in their STT APIs. But nobody has built a structured pipeline that preserves prosodic metadata as parseable markup through the entire reasoning chain. The IML-as-intermediary-format concept is genuinely novel. Whether that structured approach beats end-to-end neural methods in practice is the open question Intent Engine hasn't answered yet.

**Value prop in one sentence:**

An orchestration layer that preserves vocal emotion and intent as structured markup through the entire voice AI pipeline, from speech recognition through LLM reasoning to emotional speech synthesis.

**Verdict:** Sound -- with a critical dependency caveat. The concept is well-defined and targets a real gap. The risk is that the entire value proposition depends on `prosody_protocol` (v0.1.0-alpha) delivering accurate prosody extraction from real audio. Intent Engine is an orchestration layer built on top of unproven signal extraction. If `ProsodyAnalyzer` can't reliably distinguish sarcasm from sincerity in real-world audio, the constitutional filter, the prosody-aware prompts, and the emotional TTS all operate on bad data. No amount of clean orchestration code fixes garbage-in.

---

### EXECUTION ASSESSMENT

**Architecture:**

The three-layer pipeline design is sound. `engine.py` (645 lines) implements a clean 8-step pipeline in `process_voice_input()` (lines 160-286): validate path -> check cache -> STT transcribe -> prosody analysis -> IML assembly -> IML validation -> emotion classification -> build Result. Each step is isolated with appropriate error handling. The provider-agnostic adapter pattern (abstract base class + lazy-loaded concrete adapters + factory function) is applied consistently across STT (`stt/base.py`), LLM (`llm/base.py`), and TTS (`tts/base.py`).

The Prosody Protocol delegation is correct throughout. `engine.py:21-33` imports `ProsodyAnalyzer`, `IMLAssembler`, `IMLParser`, `IMLValidator`, `RuleBasedEmotionClassifier` directly from `prosody_protocol` and never reimplements them. This is the right call.

**Code quality:**

High for what it is. Type annotations are comprehensive, `pyproject.toml:69-70` enforces `mypy --strict`. Error handling follows a clear hierarchy (`errors.py`): `IntentEngineError` -> `STTError`/`LLMError`/`TTSError`. The constitutional filter (`constitutional/evaluator.py`) implements a tight 4-branch decision tree (lines 183-252) with clean separation between rule parsing (`rules.py`), evaluation logic (`evaluator.py`), and the filter interface (`filter.py`).

The prosody-aware system prompt (`llm/prompts.py:16-143`) is the standout artifact. It teaches LLMs the full IML tag set with concrete examples, includes a prosody-to-intent mapping table, and enforces structured JSON responses. Prompt version tracking (`PROMPT_VERSION = "1.0.0"`) is a thoughtful touch. This is cheap to test and high-impact.

**What's actually been validated:**

Nothing with real audio. The 618 tests mock all providers. `ProsodyAnalyzer`, `IMLAssembler`, STT adapters, LLM adapters, and TTS adapters are all faked in tests. The performance metrics in the README (`README.md:583-599`) are explicitly labeled as "design targets, not yet validated by benchmark runs." There is zero evidence the pipeline has processed a real audio file end-to-end.

**Over-engineering indicators:**

1. `CloudEngine` (`cloud_engine.py`, 386 lines) implements retry logic with exponential backoff, rate limiting, and session management for `https://api.intentengine.ai/v1` -- a service that doesn't exist. The README acknowledges this: "The managed cloud service is not yet available" (`README.md:518`). This is 386 lines of dead code.

2. Three deployment modes (`CloudEngine`, `HybridEngine`, `LocalEngine`) for zero production users. Each adds test surface, documentation burden, and maintenance overhead. The base `IntentEngine` hasn't been validated.

3. The emotion-to-voice parameter mapping table (`tts/base.py:37-90`) specifies precise dB offsets and rate multipliers for 13 emotions (e.g., empathetic: -5% pitch, 0.9x rate, -2dB). These numbers appear theoretical, not empirically tuned. The `style_notes` field ("Warm, slightly slower") is more honest than the specific numbers.

4. Four platform integration adapters (`integrations/twilio.py`, `slack.py`, `discord.py`, `server.py`) for a beta SDK. These are thin scaffolds that demonstrate breadth but add maintenance cost.

5. A training pipeline (`training/fine_tuner.py`, `evaluator.py`, `mavis_converter.py`) that wraps Prosody Protocol tools. No training data exists. No models have been fine-tuned. This is speculative infrastructure.

**Known code issues from the previous evaluation (still present):**

- Sync wrappers at `engine.py:621-644` use `asyncio.run()` which is correct but creates a new event loop per call -- problematic if called from an existing async context.
- `cloud_engine.py:92-107` lazy-initializes `httpx.AsyncClient` but `httpx` is not declared as a dependency anywhere in `pyproject.toml`.
- The LRU cache (`engine.py:136-149`) uses `OrderedDict` correctly (this was fixed since the prior review), but the SHA-256 hash computation at `engine.py:152-158` reads the entire audio file into memory for cache keying, which could be expensive for large files.

**Verdict:** Over-engineered relative to validation stage. The code quality is genuinely high -- well-typed, well-documented, cleanly architectured. But the engineering effort has been distributed across deployment modes, platform integrations, and training infrastructure instead of concentrated on proving the core pipeline works with real audio. This is a common pattern: building production-grade scaffolding around an unvalidated core.

---

### SCOPE ANALYSIS

**Core Feature:** The prosody-preserving pipeline: `IntentEngine.process_voice_input()` -> `generate_response()` -> `synthesize_speech()`. This 8-step pipeline in `engine.py:160-364` is the one thing that justifies the project's existence.

**Supporting:**
- STT provider abstraction (`stt/base.py`, adapters) -- necessary for audio input
- LLM provider abstraction (`llm/base.py`, adapters) with prosody-aware system prompt (`llm/prompts.py`) -- necessary for interpretation
- TTS provider abstraction (`tts/base.py`, adapters) with emotion-to-voice mapping -- necessary for output
- Constitutional filter (`constitutional/`) -- directly supports the "verify genuine intent" value proposition; this is the most differentiated feature
- Accessibility profile management (`engine.py:527-617`) -- supports the "atypical prosody" use case

**Nice-to-Have:**
- `HybridEngine` (`hybrid_engine.py`) -- useful deployment mode but not needed until core is proven
- `LocalEngine` (`local_engine.py`) -- sovereignty use case, but premature before validation
- LRU cache for audio results (`engine.py:136-149`) -- minor performance optimization
- `type_to_speech` AAC method (`engine.py:486-523`) -- accessibility feature, important but secondary

**Distractions:**
- `integrations/twilio.py` -- surface-level webhook scaffold; should be an example app
- `integrations/slack.py` -- same
- `integrations/discord.py` -- same
- `integrations/server.py` -- FastAPI server useful for testing but should be a separate package or in `examples/`
- `training/fine_tuner.py` -- wraps Prosody Protocol's tools without adding value; no training data exists
- `training/mavis_converter.py` -- Mavis bridge is a Prosody Protocol concern
- `training/evaluator.py` -- same

**Wrong Product:**
- `CloudEngine` (`cloud_engine.py`, 386 lines) -- this is a REST client for a managed service that doesn't exist. It is a separate product (the SaaS offering). Ship it as `intent-engine-cloud` when the infrastructure exists. Right now it's dead code in the core library.

**Scope Verdict:** Feature Creep. The core pipeline + constitutional filter is focused and compelling. But the project has expanded to 38 source files, 42 test files, and 618 tests for a beta SDK that hasn't been validated with real audio. The breadth (3 deployment modes, 4 platform integrations, training pipeline, accessibility profiles) is impressive for a demo but counterproductive for a library trying to prove its core thesis.

---

### RECOMMENDATIONS

**CUT:**
- `intent_engine/cloud_engine.py` and its tests -- dead code for a non-existent service
- `intent_engine/training/` (all 3 files and tests) -- thin wrappers around Prosody Protocol with no training data; reintroduce when models exist
- `intent_engine/integrations/twilio.py`, `slack.py`, `discord.py` and their tests -- move to `examples/` directory

**DEFER:**
- `HybridEngine` and `LocalEngine` -- freeze development until `IntentEngine` is validated with real audio
- `integrations/server.py` -- keep but move to a separate `intent-engine-server` package or `examples/`
- The README's roadmap section (`README.md:743-768`) -- remove specific quarter/year timelines until the core is proven

**DOUBLE DOWN:**
- **End-to-end validation with real audio.** This is the single most important missing piece. Record 20-30 test utterances covering sarcasm, frustration, joy, uncertainty, and calm. Run them through the full pipeline. Publish actual accuracy numbers alongside the current "design targets." This is the difference between a concept and a product.
- **The constitutional filter.** `constitutional/evaluator.py` implements a genuinely novel idea: using prosodic features to verify intent before executing sensitive actions. This is the most defensible competitive advantage. Expand the rule language, add more prosody conditions, and build a compelling demo showing it catching sarcastic "delete everything" commands.
- **The prosody-aware LLM prompt.** `llm/prompts.py` is strong work. Test it: send IML-annotated text to Claude and GPT-4, measure whether they correctly interpret sarcasm, urgency, and emotion from the markup. This is cheap (just API calls) and would produce concrete evidence that the approach works.
- **Honest documentation.** The README (`README.md`, 867 lines) mixes established facts with aspirational claims. The accuracy metrics table, the pricing section, the cloud service references, and the hardware tier recommendations all present speculation as reality. Distinguish between "what works today" and "what we're building toward."

**FINAL VERDICT:** Refocus

The concept is sound, the code is well-written, and the constitutional filter is genuinely novel. But the project has spread too thin. Cut the dead weight (CloudEngine, training wrappers, platform integration scaffolds), stop building deployment modes for zero users, and focus the next iteration entirely on proving the core thesis: that carrying structured prosodic metadata through a voice AI pipeline produces measurably better outcomes than treating speech as flat text. One end-to-end demo with real audio and real accuracy numbers is worth more than 618 passing mock tests.

**Next Step:** Record 10 audio clips with clear emotional signals (3 sarcastic, 3 frustrated, 2 calm, 2 joyful). Run each through `IntentEngine.process_voice_input()` with real (unmocked) Prosody Protocol components. Compare detected emotions against ground truth. Publish results.
