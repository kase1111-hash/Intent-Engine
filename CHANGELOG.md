# Changelog

All notable changes to Intent Engine are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The package is not
published to PyPI and its version is still `0.8.0`; everything below is
unreleased and lists what changes for a user upgrading from `0.8.0` as it was
before these fixes.

## [Unreleased]

The project could not be installed or imported as documented, CI had failed on
every recent run, and an audit found that the emotion pipeline, prosody
profiles and the constitutional filter did not behave as documented against
Prosody Protocol 0.1.0a3 (a breaking upstream release). This release fixes
those, and hardens what an adversarial review of the fixes found.

### Breaking changes

**Dependencies and install**
- Requires `prosody-protocol[audio]>=0.1.0a3`. Prosody Protocol is not on PyPI:
  install it from GitHub first (the commit CI pins is in
  `.github/workflows/ci.yml`), then this package. A bare `prosody-protocol`
  install no longer works because the audio stack moved into its `[audio]`
  extra.
- Extras now name versions the adapters can work with: `deepgram-sdk>=5,<8`
  (and `httpx`), `anthropic>=0.40`, `openai>=1.56`, `elevenlabs>=1.8.1`,
  `pyttsx3>=2.99` (plus the system eSpeak NG library), `openai-whisper>=20231106`.
  The `coqui` extra installs the maintained `coqui-tts` fork, because `TTS`
  0.22 cannot install on Python 3.12+. Dev floors: `ruff>=0.13`.
- New `examples` extra for the example integrations.

**Emotion and profiles**
- `Result.emotion`, `confidence` and `suggested_tone` come from the assembled
  IML. They used to be `("neutral", 0.0, "neutral")` for every recording.
  `("neutral", 0.0)` now means *no emotion was reported*: a single utterance
  without calibration speech cannot report one. `suggested_tone` describes the
  user; the tone for the reply is `Response.emotion`.
- Prosody profiles must use the Prosody Protocol vocabulary (`pitch`,
  `pitch_contour`, `volume`, `rate`, `quality`, `pause_frequency`,
  `emphasis_frequency`). The old private keys (`f0_mean`, `intensity_mean`,
  `speech_rate`, absolute Hz/dB thresholds) raise `ProfileError`. Profiles are
  validated when loaded or set, are applied per utterance against the speaker's
  baseline, and show up as `x-profile` in the IML. `create_profile` defaults to
  version `1.0.0`.

**Constitutional filter** (safer, and stricter)
- Fails closed: an unknown emotion (an abstention, or confidence below
  `min_emotion_confidence`, default 0.5) fails a required emotion list; a
  required `pitch_variance` or `speaking_rate` that could not be measured
  fails; the most restrictive decision wins (deny, then `two_factor`, then
  `explicit_confirmation`, then allow), whatever the order of the rules.
- `IntentEngine.evaluate_result(intent, result)` is the gate. It weighs every
  emotion reported for the turn: a forbidden emotion in any sentence denies, and
  every reported emotion must satisfy a required list. `evaluate_intent` and
  `ConstitutionalFilter.evaluate` gain keyword-only `emotion_confidence` and
  `min_emotion_confidence`; without a confidence they read an abstention as a
  measured `neutral`.
- Triggers match whole word sequences after normalising case and separators
  (`delete all` matches `delete_all_files`; `payments` no longer matches
  `payment`; `end` no longer matches `send_money`). List every phrasing you need.
- The rules schema is strict: unknown keys, typos, empty sections, duplicate
  rule names, `verification` without `required_prosody`, and an empty condition
  raise `ValueError` naming the file and rule. `pitch_variance` is measured in
  semitones. Rules hold tuples; emotion labels compare case-insensitively.
- `constitutional_rules=""` (for example an unset environment variable) and
  `prosody_profile=""` raise `ValueError` instead of silently running without
  the filter or profile. `pathlib.Path` is accepted for both.
- `denial_reason` and logs no longer contain the detected emotion.

**Engine**
- Cache hits return copies; the key includes the active profile; `cache_size<=0`
  disables the cache (it raised `KeyError`); `clear_cache()` and `close()` are new.
- The `*_sync` wrappers share one background event loop per engine and raise
  `RuntimeError` from a running loop. Prosody-analysis failures fall back to
  text-only IML (previously the documented fallback never triggered).
- `LocalEngine`/`HybridEngine` pass model options to the parameters the
  adapters actually take (`stt_model`, `tts_model` and `llm_model` were silently
  ignored), raise `ValueError`/`FileNotFoundError` for impossible combinations,
  and compute `is_fully_local`/`is_llm_local`. `HybridEngine` gains
  `validate_models` (default on: a missing llama.cpp file fails at construction).
- `Response` gains `intent`. `type_to_speech` sends plain text to providers
  without SSML support.

**Adapters**
- `ClaudeLLM` no longer sends `temperature` by default (current SDKs reject it;
  a value you set is sent through `extra_body`), and its default `max_tokens` is
  larger. Its default model id is unchanged and deprecated upstream: set `model=`
  explicitly.
- LLM adapters open a client per call, parse replies strictly (one JSON object;
  fenced or padded JSON tolerated), raise `LLMError` (never `KeyError`), map
  `suggested_emotion` onto the 13 core emotions, and put no reply text in error
  messages. The system prompt was rewritten (`PROMPT_VERSION` 1.1.0): every IML
  example validates.
- `DeepgramSTT` was rewritten for `deepgram-sdk` 5-7. STT adapters build word
  timings with `prosody_protocol.alignment` and raise `STTError`.
- `ESpeakTTS`: `volume` now anchors the loudest emotion, so default neutral
  speech is about 6 dB quieter, and empty output raises `TTSError` instead of
  returning an empty WAV. `ElevenLabsTTS` reports the real format and sample
  rate. Emotion lookup never raises.
- Example integrations moved to `examples/integrations/{twilio_voice,slack_bot,
  discord_bot,rest_server}.py` (import as `examples.integrations.<module>`); the
  REST server is started with `uvicorn --factory`, and an empty `INTENT_API_KEY`
  fails closed.

### Fixed
- `import intent_engine` failed on a fresh install; `pip install` could not
  resolve Prosody Protocol.
- `ClaudeLLM` raised `TypeError` on every call with current SDKs; `DeepgramSTT`
  failed on every SDK version its extra allowed; the cached async clients failed
  with "Event loop is closed" on every second `*_sync` call.
- eSpeak returned an empty WAV for text over about 150 characters; the llama.cpp
  path and eSpeak/Whisper/Coqui/AssemblyAI calls blocked the event loop.
- A `*_sync` call racing `close()` could hang; `SystemExit` from a provider
  killed the shared loop.

### Security
- The constitutional filter no longer approves a sensitive action when no
  prosody or emotion could be measured, or when a forbidden emotion sits in a
  less confident sentence than another.
- Example integrations: Twilio and Slack signature checks failed open on an empty
  secret; TwiML was not escaped; upload and body limits could be bypassed; media
  downloads had no host allow-list, size cap or deadline and sent the Slack
  token anywhere; errors leaked internals; signed URLs reached logs.
- Emotion labels, intents and reply text are kept out of INFO-and-above logs.

### Packaging and CI
- Wheel metadata declares `Apache-2.0`; the build needs `hatchling>=1.27`.
- CI installs Prosody Protocol from GitHub at a pinned commit, lints `examples/`,
  and a new `sdk-contracts` job runs the whole suite against the real provider
  SDKs and the real eSpeak library. The suite no longer depends on proxy
  settings or CPU load.

### Migration notes
- Rewrite prosody profiles to the Prosody Protocol vocabulary (see the README's
  Prosody Profiles section) and validate them with `ProfileLoader`.
- Replace `evaluate_intent(intent, feats, result.emotion)` with
  `evaluate_result(response.intent, result)`. Expect more verification requests:
  a single sentence reports no emotion, so an action guarded by a required
  emotion list now asks for verification.
- Check your rules files against the strict schema and list every phrasing of an
  intent in `triggers`.
- Set `model=` for `ClaudeLLM`, and replace `stt_model`/`tts_model` typos that
  used to be ignored. Pass `validate_models=False` to `HybridEngine` if the model
  file is mounted after construction.
