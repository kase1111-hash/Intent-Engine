# Intent Engine -- Comprehensive Software Evaluation

**Evaluator:** Claude (Opus 4.6)
**Date:** 2026-02-06
**Commit evaluated:** `be2335b670e6affec100f93a9e9b7fa043bfa3b1` (HEAD at time of evaluation)
**Repository:** <https://github.com/kase1111-hash/Intent-Engine>

---

## Evaluation Parameters

| Parameter | Value |
|---|---|
| **Strictness** | STANDARD |
| **Context** | LIBRARY-FOR-OTHERS |
| **Purpose Context** | IDEA-STAKE + ECOSYSTEM-COMPONENT |
| **Focus Areas** | concept-clarity-critical, prosody-protocol-alignment |

---

## EXECUTIVE SUMMARY

**Overall Assessment: NEEDS-WORK**
**Purpose Fidelity: ALIGNED**
**Confidence Level: HIGH**

Intent Engine is a conceptually clear and well-documented idea-stake for prosody-aware voice AI. The project occupies a distinct conceptual territory: the orchestration layer that wires IML-based prosody analysis (from the Prosody Protocol SDK) into a full STT -> LLM -> TTS pipeline with constitutional safety filtering and accessibility profiles. The implementation faithfully follows its own spec and execution guide across all 10 phases. Code quality is consistent, the architecture is clean, and the provider adapter pattern is well-executed. However, the project has several gaps that prevent a PRODUCTION-READY rating: the `CLAUDE.md` states "no implementation code exists yet" (now outdated), the README contains aspirational claims (accuracy metrics, pricing, SaaS service) that are not validated by the current codebase, and the sync wrappers use the deprecated `get_event_loop()` pattern. These are addressable issues, not fundamental flaws.

---

## SCORES (1--10)

| Dimension | Score | Justification |
|---|---|---|
| **Purpose Fidelity** | **8** | Implementation closely follows spec/EXECUTION_GUIDE. Minor doc drift (CLAUDE.md says "no code exists"). README metrics are aspirational, not measured. |
| -- Intent Alignment | 8 | All specified features are implemented. No significant scope creep. |
| -- Conceptual Legibility | 9 | A reader can grasp the core idea within 2 minutes. README leads with the problem/idea. Module structure mirrors the concept. |
| -- Specification Fidelity | 8 | Behavior matches documented behavior. Identifiers match spec terminology. Some API method signatures diverge slightly from spec examples. |
| -- Doctrine of Intent | 7 | Provenance chain is clear. Git history shows phased development. Timestamps establish priority. AI-generated nature is self-evident from rapid commit cadence. |
| -- Ecosystem Position | 9 | Clear non-overlapping territory vs. Prosody Protocol. Dependency is accurate and justified. |
| **Implementation Quality** | **7** | Clean, readable code with good patterns. Some issues: deprecated async patterns, hand-rolled LRU cache, broad exception catches, Claude client re-created per call. |
| **Resilience & Risk** | **6** | Good error hierarchy. Graceful prosody fallback. But: no input sanitization on audio paths, temp files in server could leak, Claude adapter doesn't validate JSON before parse, no rate limiting on REST server. |
| **Delivery Health** | **7** | pyproject.toml, CI/CD, pre-commit, Makefile all present. 571 tests. 80% coverage gate. Missing: no CONTRIBUTING.md (referenced in README), no LICENSE file, no examples/ directory, docs reference non-existent files. |
| **Maintainability** | **8** | Clean module boundaries. Provider-agnostic architecture makes extension easy. Low coupling between layers. Onboarding aided by comprehensive docs. |
| **Overall** | **7.3** | |

---

## FINDINGS

### I. Purpose Drift Findings (spec vs. code)

**PD-1: CLAUDE.md says "no implementation code exists" (MODERATE)**
- File: `CLAUDE.md:7` -- states "Status: Beta (v0.8.0) - specification/design phase. No implementation code exists yet"
- Reality: 38 source files, 37 test files, 7,340 lines of Python exist. This is stale documentation that contradicts the current state.

**PD-2: README claims unvalidated accuracy metrics (MODERATE)**
- File: `README.md:588-594` -- states "Emotion Detection: 87%", "Sarcasm Detection: 82%", etc.
- Reality: No benchmark results exist in the repo. No trained models are present. These metrics are aspirational targets from `spec.md:384-392`, not measured outcomes. They read as established facts in the README.

**PD-3: README presents a non-existent managed cloud service (MODERATE)**
- File: `README.md:519-535` -- references `intent-engine.io`, API keys, pricing tiers
- Reality: No cloud service infrastructure exists. `CloudEngine` is a client stub. Pricing is speculative.

**PD-4: Spec `Result` has `intent` field but implementation defaults to `None` (LOW)**
- File: `spec.md:235` documents `intent: str` as a Result field. Implementation at `intent_engine/models/result.py:40` makes it `str | None = None`.
- The `process_voice_input` pipeline never populates this field; it requires a separate `generate_response` call. The spec doesn't clarify this two-step flow.

**PD-5: README API examples differ from actual implementation signatures (LOW)**
- File: `README.md:177-208` shows `engine = IntentEngine(stt_provider="whisper-prosody", ...)` which matches. But `result.iml` in the README is used as a string directly, while the actual `Result` also has `iml_document` (the parsed object). Not a breaking divergence, but the README doesn't mention the richer API.

**PD-6: EXECUTION_GUIDE Phase 11 (Fine-Tuning) not implemented (LOW)**
- File: `EXECUTION_GUIDE.md:827-860` defines Phase 11 for fine-tuning pipeline.
- This is explicitly marked as "Advanced" and "Optional for initial release". Not a violation, but worth noting.

### II. Conceptual Clarity Findings

**CC-1: Excellent conceptual framing (POSITIVE)**
- The README's problem/solution framing ("tone-deaf" voice AI) immediately communicates the novel idea.
- The three-layer architecture diagram is self-explanatory.
- Module structure mirrors the conceptual model perfectly: `stt/`, `llm/`, `tts/`, `constitutional/`, `integrations/`.

**CC-2: Clear dependency boundary with Prosody Protocol (POSITIVE)**
- CLAUDE.md Table ("What Prosody Protocol Provides" vs. "What Intent Engine Adds") is the strongest artifact in the entire repo for establishing conceptual territory.

**CC-3: The "why" is explicit throughout (POSITIVE)**
- Every module docstring explains its role relative to the spec.
- The EXECUTION_GUIDE provides clear rationale for every phase.

### III. Critical Findings (must fix)

**C-1: Deprecated `asyncio.get_event_loop()` in sync wrappers**
- Files: `intent_engine/engine.py:516,620,631,639`, `intent_engine/cloud_engine.py:354,365,374`
- `asyncio.get_event_loop().run_until_complete()` is deprecated since Python 3.10 and raises `DeprecationWarning` in 3.12+. Should use `asyncio.run()` or a proper sync bridge pattern (e.g., `anyio.from_thread.run`).

**C-2: ClaudeLLM creates a new `AsyncAnthropic` client per call**
- File: `intent_engine/llm/claude.py:83` -- `client = anthropic.AsyncAnthropic(api_key=self._api_key)` inside `interpret()`.
- This creates a new HTTP connection pool per request. The client should be instantiated once in `__init__` and reused. Same issue likely exists in `OpenAILLM`.

**C-3: No JSON parse error handling in Claude adapter**
- File: `intent_engine/llm/claude.py:94` -- `parsed = json.loads(raw_text)` with no try/except.
- If Claude returns non-JSON (network error, safety refusal, unexpected format), this will raise an unhandled `json.JSONDecodeError` that propagates as-is instead of as an `LLMError`.

### IV. High-Priority Findings

**H-1: Hand-rolled LRU cache should use `functools.lru_cache` or `OrderedDict`**
- File: `intent_engine/engine.py:136-153`
- The manual cache implementation uses `list.remove()` which is O(n). Python's `functools.lru_cache` or `collections.OrderedDict` provide correct, performant LRU semantics. At `cache_size=128` the performance impact is negligible, but the hand-rolled approach is more error-prone.

**H-2: Broad `except Exception` hides specific errors**
- File: `intent_engine/engine.py:220` -- catches all exceptions from prosody analysis, including `KeyboardInterrupt` (which is `BaseException`, so actually fine here), but more importantly hides programming errors like `AttributeError` or `TypeError` that indicate bugs, not transient failures.
- Same pattern at `intent_engine/engine.py:260` (profile application).

**H-3: REST server temp file handling could leak on crash**
- File: `intent_engine/integrations/server.py:119-155`
- Uses `tempfile.NamedTemporaryFile(delete=False)` with cleanup in `finally`. If the process is killed between creation and the `finally` block, files leak. Consider using a context manager or configuring a tmp directory with periodic cleanup.

**H-4: No input validation on `audio_path` parameter**
- File: `intent_engine/engine.py:164` -- `audio_path` is passed directly to `open()` for hashing and to providers.
- No validation that the path exists, is a file (not a directory), or is within expected bounds. While providers like `WhisperSTT` validate path existence (`whisper.py:76-77`), the engine itself doesn't, and the hash computation at line 159 would raise a generic `FileNotFoundError` or `IsADirectoryError`.

**H-5: Missing LICENSE file and CONTRIBUTING.md**
- File: `README.md:807` says "See [LICENSE](./LICENSE)" but no LICENSE file exists.
- File: `README.md:799` says "See [CONTRIBUTING.md](./CONTRIBUTING.md)" but no such file exists.
- These are broken references in the primary documentation.

### V. Moderate Findings

**M-1: ElevenLabs adapter creates a synchronous client in an async method**
- File: `intent_engine/tts/elevenlabs.py:108` -- `ElevenLabsClient(api_key=...)` is the sync client, used inside `async def synthesize()`. The `client.text_to_speech.convert()` call at line 110 is synchronous and blocks the event loop. Should use an async client or run in an executor.

**M-2: `type_to_speech` doesn't use the IML prediction result**
- File: `intent_engine/engine.py:507-508` -- calls `predictor.predict(text, context=emotion)` but discards the return value, then synthesizes from the original `text` and `emotion`. The IML prediction is wasted.

**M-3: `STT_PROVIDERS` dict stores string paths instead of classes**
- File: `intent_engine/stt/__init__.py:48-52` -- `STT_PROVIDERS` maps names to import path strings, but `_get_provider_class()` uses `if/elif` chains for actual lazy import. The dict is only used for error messages. This dual tracking is mildly confusing; a single mechanism would be cleaner.

**M-4: Module-level `app` global in server.py is fragile**
- File: `intent_engine/integrations/server.py:28-29,188-189` -- `app: Any = None` at module level, mutated by `create_app()`. This is for `uvicorn` compatibility but creates implicit shared state. A factory pattern is already in place; the global is redundant complexity.

**M-5: YAML rule parser doesn't validate emotion names against PP vocabulary**
- File: `intent_engine/constitutional/rules.py:84-174` -- accepts any string as an emotion in rules. No validation against the Prosody Protocol core emotion vocabulary. Typos in rule files (e.g., `"frsutrated"`) would silently never match.

**M-6: CloudEngine assumes specific JSON response shapes**
- File: `intent_engine/cloud_engine.py:228-248` -- `data["text"]`, `data["iml"]`, etc. with no validation. If the cloud API evolves or returns unexpected shapes, these will raise `KeyError` instead of meaningful errors.

**M-7: `py.typed` marker present but mypy strict mode may not fully pass**
- File: `intent_engine/py.typed` exists, `pyproject.toml:69-70` sets `strict = true`.
- Several `Any` type annotations exist (e.g., `intent_engine/integrations/server.py:33`), and external mock-heavy patterns in tests may trigger mypy issues. No evidence in the repo that `mypy --strict` actually passes cleanly.

### VI. Observations (non-blocking)

**O-1:** The Prosody Protocol is imported successfully in `conftest.py` and throughout the codebase, confirming the dependency contract works.

**O-2:** The `EMOTION_VOICE_MAP` in `tts/base.py` covers all 13 core emotions from the Prosody Protocol vocabulary. Well-aligned.

**O-3:** The commit history shows a clean, phased progression matching the EXECUTION_GUIDE. Each phase adds tests alongside implementation. This is good engineering practice.

**O-4:** The CI pipeline includes `pip-audit` for dependency security scanning. This is above-average for a beta project.

**O-5:** The `prosody-aware system prompt` in `llm/prompts.py` is unusually well-crafted -- it teaches the LLM the full IML tag set with examples and includes the prosody-to-intent mapping table. Prompt version tracking (`PROMPT_VERSION = "1.0.0"`) is a thoughtful detail.

**O-6:** Test coverage across modules is well-organized with the directory structure mirroring the source. The `conftest.py` provides reusable factory functions and the IML validation gate.

**O-7:** The `HybridEngine` and `LocalEngine` are clean thin subclasses of `IntentEngine` that override defaults and add model validation. Minimal code duplication.

---

## POSITIVE HIGHLIGHTS

1. **Strong conceptual clarity.** The README's problem/solution framing, the CLAUDE.md dependency boundary table, and the spec are all well-crafted artifacts that establish the idea clearly and defensibly.

2. **Clean architectural boundaries.** The provider adapter pattern (abstract base class + lazy-loaded concrete adapters + factory function) is applied consistently across STT, LLM, and TTS. This makes the system genuinely provider-agnostic.

3. **Prosody Protocol integration is correct.** The codebase faithfully delegates all IML parsing, validation, prosody analysis, and emotion classification to `prosody_protocol`. No reimplementation detected. Type references (`IMLDocument`, `SpanFeatures`, `WordAlignment`) are used throughout.

4. **Constitutional filter is well-designed.** The YAML rule schema, prosody condition evaluation, and four-branch decision logic (allow / deny / require verification / deny outright) match the spec precisely. The separation of concerns between `rules.py`, `evaluator.py`, and `filter.py` is clean.

5. **Comprehensive test suite.** 571 tests with IML contract tests, end-to-end pipeline tests, error recovery tests, and accessibility profile tests. The shared `assert_valid_iml` gate ensures IML conformance across the test suite.

6. **Accessibility is first-class.** The profile management API (`load_profile`, `set_profile`, `create_profile`, `validate_profile`), feature label derivation, and `type_to_speech` augmentative communication support demonstrate that accessibility is built into the architecture, not bolted on.

7. **Defensive lazy imports.** Provider SDKs are imported lazily on first use, with clear `ImportError` messages telling users which `pip install` extra to run. This means the core package installs cleanly without heavy ML dependencies.

---

## RECOMMENDED ACTIONS

### Immediate (Purpose)

1. **Update CLAUDE.md** to reflect that implementation is complete (remove "No implementation code exists yet" statement and "This repo currently contains only documentation"). Update the Repository Structure section to match the actual file tree.

2. **Qualify README metrics** as targets/goals, not measured outcomes. Add a note like "Performance targets (pending benchmark validation)" above the accuracy and latency tables.

3. **Remove or qualify cloud service references** in the README until infrastructure exists. Mark them as "Planned" rather than presenting them as available.

4. **Add LICENSE and CONTRIBUTING.md** files or remove the references from README.

### Immediate (Quality)

5. **Replace `asyncio.get_event_loop().run_until_complete()` with `asyncio.run()`** across all sync wrappers (`engine.py`, `cloud_engine.py`). This eliminates deprecation warnings on Python 3.12+.

6. **Fix ClaudeLLM (and likely OpenAILLM) to reuse HTTP clients** -- instantiate the API client in `__init__`, not per-call.

7. **Add JSON parse error handling** in `claude.py:94` and equivalent LLM adapters. Wrap `json.loads()` in a try/except that raises `LLMError` with context.

8. **Fix `type_to_speech` to use the IML prediction result** or remove the unused `TextToIML` call.

### Short-term

9. **Replace the hand-rolled LRU cache** with `collections.OrderedDict` or a lightweight library like `cachetools`.

10. **Add emotion vocabulary validation** to the constitutional YAML rule parser to catch typos.

11. **Fix the ElevenLabs adapter** to either use an async client or offload the sync call to a thread executor.

12. **Add path validation** in `IntentEngine.process_voice_input()` before attempting the file hash.

13. **Verify that `mypy --strict` passes** cleanly and document the result. Fix any type issues.

### Long-term

14. **Build actual benchmark infrastructure** using `prosody_protocol.Benchmark` (as described in Phase 10 of the EXECUTION_GUIDE) and publish measured accuracy results.

15. **Add integration tests with real audio files** (even small synthetic ones) to validate the full pipeline beyond mocks.

16. **Consider Docker or Makefile targets for local development** to simplify onboarding.

17. **Implement Phase 11 (Fine-Tuning Pipeline)** when a trained model is available.

---

## QUESTIONS FOR AUTHORS

1. **Has `mypy --strict` been run against the full codebase?** The `pyproject.toml` enables strict mode, but the CI pipeline runs `mypy intent_engine/` without `--strict` flag explicitly. Are there known type errors?

2. **What is the plan for the LICENSE file?** The README says "tbd" for the license. Is Apache 2.0 the intended choice given the badge?

3. **Are the accuracy metrics in the README from any preliminary experiments**, or are they purely aspirational targets from the spec? If preliminary, where is the benchmark data?

4. **Is the `CloudEngine` intended to be functional**, or is it a forward-looking API stub? Its `_DEFAULT_BASE_URL` points to `https://api.intentengine.ai/v1` which does not exist.

5. **Has the end-to-end pipeline been tested with real audio and real Prosody Protocol analysis** (not mocked)? The test suite mocks `ProsodyAnalyzer`, `IMLAssembler`, etc. -- do you have evidence that the real components integrate correctly?

6. **What is the relationship between this repo's author and the Prosody Protocol repo?** They share the same GitHub org (`kase1111-hash`). Is this the same author establishing a two-project ecosystem?

---

*Evaluation generated on 2026-02-06 by Claude Opus 4.6.*
*Evaluated commit: `be2335b670e6affec100f93a9e9b7fa043bfa3b1`*
