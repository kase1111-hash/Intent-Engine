"""IntentEngine -- main orchestrator.

Wires STT, prosody analysis, LLM interpretation, constitutional
filtering, and TTS synthesis into a single coherent pipeline.

Uses Prosody Protocol components (ProsodyAnalyzer, IMLAssembler,
IMLParser, IMLValidator, RuleBasedEmotionClassifier) for all
IML-related operations.  Intent Engine adapters (STT, LLM, TTS)
handle provider-specific communication.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import operator
import os
import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from dataclasses import replace
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from prosody_protocol import (
    IMLAssembler,
    IMLDocument,
    IMLParser,
    IMLValidator,
    PauseInterval,
    ProfileError,
    ProfileLoader,
    ProsodyAnalyzer,
    ProsodyMapping,
    ProsodyProfile,
    ProsodyProtocolError,
    RuleBasedEmotionClassifier,
    SpanFeatures,
    ValidationResult,
    WordAlignment,
)

from intent_engine.constitutional.evaluator import most_restrictive, resolve_emotion
from intent_engine.constitutional.filter import ConstitutionalFilter
from intent_engine.errors import IntentEngineError, LLMError, STTError, TTSError
from intent_engine.llm import LLMProvider, create_llm_provider
from intent_engine.models.audio import Audio
from intent_engine.models.decision import Decision
from intent_engine.models.response import Response
from intent_engine.models.result import Result
from intent_engine.stt import STTProvider, create_stt_provider
from intent_engine.tts import TTSProvider, create_tts_provider

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_T = TypeVar("_T")


def _path_option(name: str, value: object, what: str, none_means: str) -> str:
    """Return a path-valued constructor option as a string, or raise if it is unusable.

    ``None`` (the option is unset) is handled by the caller.  An empty string
    is an error rather than "unset": an unset environment variable expands to
    ``""``, and silently running without a safety filter or an accessibility
    profile beats nothing.
    """
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(
                f"{name} must be a path to {what}, not an empty string "
                f"(pass None {none_means})"
            )
        return value
    if isinstance(value, os.PathLike):
        path = os.fspath(value)
        if isinstance(path, str):
            return path
    raise TypeError(
        f"{name} must be a str or os.PathLike path to {what} (or None {none_means}), "
        f"got {type(value).__name__}"
    )


class _LoopThread:
    """An event loop running in a daemon thread, for the ``*_sync`` wrappers.

    Provider SDKs cache async clients (connection pools) that stay bound
    to the loop they were first used on, so every synchronous call has to
    run on the same loop rather than on a fresh ``asyncio.run`` each time.
    The thread stops when *owner* is garbage collected.
    """

    def __init__(self, owner: object) -> None:
        self.pid = os.getpid()
        self.loop = asyncio.new_event_loop()
        self._stopping = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="intent-engine-sync", daemon=True
        )
        self.thread.start()
        self._finalizer = weakref.finalize(owner, self._request_stop)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            while not self._stopping.is_set():
                try:
                    self.loop.run_forever()
                except (KeyboardInterrupt, SystemExit):
                    # A provider coroutine raised it: asyncio has already set it
                    # on that call's task, so the caller of the *_sync wrapper
                    # receives it.  It must not end the loop the other calls
                    # (and later ones) share.
                    continue
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

    def _request_stop(self) -> None:
        self._stopping.set()
        with contextlib.suppress(RuntimeError):  # the loop is already closed
            self.loop.call_soon_threadsafe(self.loop.stop)

    def close(self) -> None:
        """Stop the loop and wait for its thread.

        In a forked child, where the thread does not exist, this only
        drops the reference.
        """
        self._finalizer.detach()
        if self.pid == os.getpid():
            self._request_stop()
            self.thread.join()


class IntentEngine:
    """Main orchestrator for the prosody-aware AI pipeline.

    Coordinates the full pipeline: STT transcription, prosody
    analysis, IML assembly/validation, LLM interpretation,
    constitutional filtering, and TTS synthesis.

    Parameters
    ----------
    stt_provider:
        STT provider name (``"whisper-prosody"``, ``"deepgram"``,
        ``"assemblyai"``).
    llm_provider:
        LLM provider name (``"claude"``, ``"openai"``, ``"local"``).
    tts_provider:
        TTS provider name (``"elevenlabs"``, ``"coqui"``, ``"espeak"``).
    constitutional_rules:
        Optional path (``str`` or ``pathlib.Path``) to a YAML file with
        constitutional rules.  ``None`` runs without the filter; an empty
        string is a ``ValueError`` (an unset environment variable must not
        silently disable it), and any other type a ``TypeError``.
    prosody_profile:
        Optional path (``str`` or ``pathlib.Path``) to a prosody profile
        JSON for atypical prosody handling.  The profile is validated when
        the engine is created (``ProfileError`` if it is invalid).  An empty
        string is a ``ValueError``, as for ``constitutional_rules``.
    cache_size:
        Maximum number of audio results to cache (LRU), an integer; ``0`` or
        less disables caching.  Results are cached per audio content and active
        prosody profile.  They hold transcripts and detected emotion, so
        call :meth:`clear_cache` when they should not be kept.
    stt_kwargs:
        Provider-specific keyword arguments for the STT adapter.
    llm_kwargs:
        Provider-specific keyword arguments for the LLM adapter.
    tts_kwargs:
        Provider-specific keyword arguments for the TTS adapter.
    """

    def __init__(
        self,
        stt_provider: str = "whisper-prosody",
        llm_provider: str = "claude",
        tts_provider: str = "elevenlabs",
        constitutional_rules: str | os.PathLike[str] | None = None,
        prosody_profile: str | os.PathLike[str] | None = None,
        cache_size: int = 128,
        stt_kwargs: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        tts_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # Intent Engine adapters
        self._stt: STTProvider = create_stt_provider(
            stt_provider, **(stt_kwargs or {})
        )
        self._llm: LLMProvider = create_llm_provider(
            llm_provider, **(llm_kwargs or {})
        )
        self._tts: TTSProvider = create_tts_provider(
            tts_provider, **(tts_kwargs or {})
        )

        # Guards the active profile/assembler pair and the result cache,
        # which concurrent tasks and threads (the *_sync wrappers) share
        self._lock = threading.Lock()

        # Prosody Protocol components
        self._analyzer = ProsodyAnalyzer()
        self._parser = IMLParser()
        self._validator = IMLValidator()
        # One classifier serves both: the assembler measures each utterance
        # against the speaker's baseline with it, and Result reports what
        # the assembler concluded (see _document_emotion).
        self._emotion_classifier = RuleBasedEmotionClassifier()
        self._assembler = IMLAssembler(emotion_classifier=self._emotion_classifier)

        # Optional constitutional filter
        self._filter: ConstitutionalFilter | None = None
        if constitutional_rules is not None:
            self._filter = ConstitutionalFilter.from_yaml(
                _path_option(
                    "constitutional_rules",
                    constitutional_rules,
                    "a YAML rules file",
                    "to run without the filter",
                )
            )

        # Event loop of the *_sync wrappers, started on first use
        self._sync_runner: _LoopThread | None = None

        # LRU cache for audio processing results
        try:
            self._cache_size = operator.index(cache_size)
        except TypeError:
            raise TypeError(
                f"cache_size must be an integer, got {type(cache_size).__name__}"
            ) from None
        self._cache: OrderedDict[str, Result] = OrderedDict()
        # Counts clear_cache() calls, so a call in flight at the time of a
        # clear does not put its result back
        self._cache_generation = 0

        # Profile loader (always available for the management API)
        self._profile_loader = ProfileLoader()

        # Optional accessibility profile (loaded from path); it is handed to
        # the assembler, which applies it per utterance
        self._profile: ProsodyProfile | None = None
        if prosody_profile is not None:
            self.set_profile(
                self.load_profile(
                    _path_option(
                        "prosody_profile",
                        prosody_profile,
                        "a profile JSON file",
                        "to run without a profile",
                    )
                )
            )

        logger.info(
            "IntentEngine initialized (stt=%s, llm=%s, tts=%s, filter=%s, profile=%s)",
            stt_provider,
            llm_provider,
            tts_provider,
            "enabled" if self._filter else "disabled",
            "enabled" if self._profile else "disabled",
        )

    def _cache_get(self, key: str) -> Result | None:
        """Retrieve a cached result by key."""
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
            return cached

    def _cache_put(self, key: str, result: Result, generation: int | None = None) -> None:
        """Store a result in the cache, evicting the oldest if full.

        With *generation* (as read before the result was computed) the result
        is dropped if :meth:`clear_cache` ran in the meantime.
        """
        if self._cache_size <= 0:
            return
        with self._lock:
            if generation is not None and generation != self._cache_generation:
                return
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self._cache_size:
                self._cache.popitem(last=False)
            self._cache[key] = result

    def clear_cache(self) -> None:
        """Drop all cached results.

        Cached results hold transcripts and detected emotion (personal
        data); call this when they should not be kept.  A call to
        :meth:`process_voice_input` that is still running when this is called
        returns its result to its caller but does not cache it.
        """
        with self._lock:
            self._cache.clear()
            self._cache_generation += 1

    @staticmethod
    def _copy_features(features: list[SpanFeatures]) -> list[SpanFeatures]:
        """Copies of *features* that share no mutable state with them.

        ``SpanFeatures`` is frozen, but its ``f0_contour`` is a plain list.
        """
        return [
            replace(f, f0_contour=None if f.f0_contour is None else list(f.f0_contour))
            for f in features
        ]

    @staticmethod
    def _audio_hash(audio_path: str) -> str:
        """Hash the *content* of an audio file for cache keying.

        The path and modification time do not matter: the same audio at
        another path is a hit, and a file overwritten in place is a miss.
        """
        h = hashlib.sha256()
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _profile_fingerprint(profile: ProsodyProfile | None) -> str:
        """Cache-key part for a profile: the result depends on it."""
        if profile is None:
            return ""
        return hashlib.sha256(repr(profile).encode("utf-8")).hexdigest()

    async def process_voice_input(
        self, audio_path: str, use_cache: bool = True
    ) -> Result:
        """Process an audio file through the full STT + prosody pipeline.

        Steps:
            1. STT transcribes audio -> text + word alignments
            2. ProsodyAnalyzer extracts features -> SpanFeatures
            3. ProsodyAnalyzer detects pauses -> PauseInterval
            4. IMLAssembler combines all into an IMLDocument
            5. IMLParser serializes the IMLDocument -> IML string
            6. IMLValidator validates the IML (asserts zero errors)
            7. Emotion and confidence are read from the assembled
               document, so ``Result`` agrees with the IML the LLM sees
            8. Return Result with all data

        The assembler classifies each utterance against the speaker's own
        baseline and leaves out emotions it is not confident about.
        Without calibration speech that needs a recording of at least
        three utterances, most of them at the speaker's usual level; a
        single sentence therefore carries no emotion.  In that case
        (and whenever the IML carries no emotion) ``Result.emotion`` is
        ``"neutral"`` with confidence ``0.0``, which means "no emotion
        reported", not "measured neutral".

        The blocking work (hashing the file, prosody analysis) runs in a
        worker thread, so other tasks on the event loop keep running while the
        audio is read and decoded.  Praat (``parselmouth``), which does most of
        the measuring, holds the GIL, though: the loop can still pause for a
        fraction of the analysis time (about half of it, so tens of
        milliseconds for a few seconds of speech and most of a second for a
        minute and a half), and concurrent recordings are analysed one after
        the other, not in parallel.  A server that handles long recordings
        should cap their length or run the engine in a separate process.

        If prosody analysis fails (audio that is unreadable, too short or
        sampled too low, for example), the turn continues with text-only
        IML: ``prosody_features`` is empty and no emotion is reported.
        Such results are not cached.  If the STT provider returns text
        without word timings, the whole recording is measured as one span
        holding the text, so the IML still carries the words (but no
        word-level prosody).

        Parameters
        ----------
        audio_path:
            Path to the audio file.
        use_cache:
            Whether to use cached results for the same audio file (and
            the same active profile).

        Returns
        -------
        Result
            Full pipeline result including IML, features, and emotion.

        Raises
        ------
        FileNotFoundError
            If *audio_path* does not exist.
        ValueError
            If *audio_path* is not a file.
        STTError
            If transcription fails.
        IntentEngineError
            If IML validation fails.
        ProsodyProtocolError
            If assembling or serializing the IML fails (for example STT
            text with control characters); upstream errors are re-raised
            as they are.
        """
        # Validate audio path
        audio = Path(audio_path)
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not audio.is_file():
            raise ValueError(f"Audio path is not a file: {audio_path}")

        # One consistent view of the profile for the whole call, so the
        # result is cached under the profile that produced it
        with self._lock:
            assembler, profile = self._assembler, self._profile
            generation = self._cache_generation

        # Check cache (results depend on the audio and on the profile)
        caching = use_cache and self._cache_size > 0
        if caching:
            audio_hash = await asyncio.to_thread(self._audio_hash, audio_path)
            cache_key = f"{audio_hash}:{self._profile_fingerprint(profile)}"
            cached = self._cache_get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for %s", audio_path)
                # a copy, so callers cannot change what later hits return
                return replace(
                    cached, prosody_features=self._copy_features(cached.prosody_features)
                )

        # Step 1: STT transcription
        try:
            transcription = await self._stt.transcribe(audio_path)
        except Exception as exc:
            raise STTError(f"STT transcription failed: {exc}") from exc

        # Steps 2-3: Prosody analysis (with fallback)
        alignments = transcription.alignments
        features: list[SpanFeatures] = []
        pauses: list[PauseInterval] = []
        degraded = False
        # Text without word timings has nothing to align to, and the IML is
        # built from alignments: the whole recording stands in as one span
        # holding the text, so the LLM still sees the words.
        untimed_text = " ".join(transcription.text.split()) if not alignments else ""
        if untimed_text:
            logger.debug("STT returned no word timings; using the whole recording as one span")
        try:
            alignments, features, pauses = await asyncio.to_thread(
                self._analyze_prosody, audio_path, alignments, untimed_text
            )
        except (ProsodyProtocolError, RuntimeError, OSError, ValueError) as exc:
            # Upstream reports audio it cannot read or analyse (unsupported
            # format, too short, sampled too low, empty) as
            # AudioProcessingError; the words are still worth answering.
            degraded = True
            logger.warning(
                "Prosody analysis failed (%s); falling back to text-only IML",
                type(exc).__name__,
            )
            logger.debug("Prosody analysis failure", exc_info=True)
            if untimed_text:
                # nothing was measured, but the words still go into the IML
                alignments = [WordAlignment(untimed_text, 0, 0)]

        # Step 4: IML assembly
        iml_doc = assembler.assemble(
            alignments,
            features,
            pauses,
            language=transcription.language,
        )

        # Step 5: Serialize to IML string
        iml_string = self._parser.to_iml_string(iml_doc)

        # Step 6: Validate IML
        validation = self._validator.validate(iml_string)
        if not validation.valid:
            error_issues = [
                i for i in validation.issues if getattr(i, "severity", "") == "error"
            ]
            if error_issues:
                raise IntentEngineError(
                    f"IML validation failed with {len(error_issues)} error(s): "
                    f"{error_issues[0]}"
                )

        # Step 7: Emotion as the assembler reported it (it also applied the
        # active prosody profile, which shows as x-profile in the IML)
        emotion, confidence = self._document_emotion(iml_doc)

        # Determine suggested tone
        suggested_tone = emotion if confidence >= 0.5 else "neutral"

        result = Result(
            text=transcription.text,
            emotion=emotion,
            confidence=confidence,
            iml=iml_string,
            iml_document=iml_doc,
            suggested_tone=suggested_tone,
            prosody_features=features,
        )

        # Cache a copy, so the caller changing the returned features is
        # harmless; a text-only fallback is not kept, in case the failure was
        # transient
        if caching and not degraded:
            self._cache_put(
                cache_key,
                replace(result, prosody_features=self._copy_features(features)),
                generation,
            )

        return result

    def _analyze_prosody(
        self, audio_path: str, alignments: list[WordAlignment], untimed_text: str = ""
    ) -> tuple[list[WordAlignment], list[SpanFeatures], list[PauseInterval]]:
        """Measure alignments' features and pauses (blocking; runs in a worker thread).

        A worker thread keeps the event loop from being blocked while audio is
        read and decoded, not while Praat measures it: Praat holds the GIL.

        With *untimed_text* (a transcript the STT gave no word timings for)
        the whole recording is measured as a single span holding it, as
        upstream does for a transcript without timings; where each word
        was said is unknown, so nothing finer, such as pauses, is placed.
        Returns the alignments to assemble along with their features and
        the pauses.
        """
        if untimed_text:
            whole = self._analyzer.analyze_recording(audio_path, text=untimed_text)
            alignments = [WordAlignment(untimed_text, whole.start_ms, whole.end_ms)]
            return alignments, [whole], []
        features = self._analyzer.analyze(audio_path, alignments)
        return alignments, features, self._analyzer.detect_pauses(audio_path)

    async def generate_response(
        self,
        iml: str,
        context: str | None = None,
        tone: str | None = None,
    ) -> Response:
        """Generate an LLM response from IML-annotated input.

        Parameters
        ----------
        iml:
            Serialized IML markup string.
        context:
            Optional conversation context.
        tone:
            Tone of the user's voice (typically ``Result.suggested_tone``),
            passed to the LLM as a hint.  It does not set the tone of the
            reply: the LLM chooses that from what the user needs (an angry
            caller may need calm), and reports it as ``Response.emotion``.
            ``"neutral"`` (what ``Result.suggested_tone`` holds when no
            emotion was reported; the classifier never reports a measured
            neutral) or an empty tone sends no hint, since the IML the model
            reads already says whether an emotion was detected.

        Returns
        -------
        Response
            Response text, the emotion to speak it with, and the intent
            the LLM parsed.

        Raises
        ------
        LLMError
            If the LLM call fails.
        """
        full_context = context or ""
        tone = (tone or "").strip()
        # "neutral" is Result.suggested_tone when the engine abstained: no
        # reading, not a measured neutral, so it must not be presented as one
        if tone and tone.lower() != "neutral":
            tone_hint = (
                f"The user's voice sounds '{tone}' (an automatic estimate that may be "
                "wrong). Choose the response tone that serves what they need; do not "
                "simply mirror it."
            )
            full_context = f"{full_context}\n{tone_hint}" if full_context else tone_hint

        try:
            interpretation = await self._llm.interpret(
                iml, context=full_context or None
            )
        except Exception as exc:
            raise LLMError(f"LLM interpretation failed: {exc}") from exc

        return Response(
            text=interpretation.response_text,
            emotion=interpretation.suggested_emotion,
            intent=interpretation.intent,
        )

    async def synthesize_speech(
        self, text: str, emotion: str = "neutral"
    ) -> Audio:
        """Synthesize speech with emotional tone.

        Parameters
        ----------
        text:
            Text to synthesize.
        emotion:
            Emotion label for voice parameter adjustment.

        Returns
        -------
        Audio
            Audio bytes and metadata.

        Raises
        ------
        TTSError
            If synthesis fails.
        """
        try:
            synthesis = await self._tts.synthesize(text, emotion=emotion)
        except Exception as exc:
            raise TTSError(f"TTS synthesis failed: {exc}") from exc

        return Audio(
            data=synthesis.audio_data,
            format=synthesis.format,
            sample_rate=synthesis.sample_rate,
            duration=synthesis.duration,
        )

    def evaluate_intent(
        self,
        intent: str,
        prosody_features: list[SpanFeatures],
        emotion: str | None = None,
        context: dict[str, object] | None = None,
        *,
        emotion_confidence: float | None = None,
        min_emotion_confidence: float = 0.5,
    ) -> Decision:
        """Evaluate an intent through the constitutional filter.

        Parameters
        ----------
        intent:
            Parsed user intent label.
        prosody_features:
            Prosody features from the pipeline.
        emotion:
            Detected emotion label.  Without *emotion_confidence* the label
            is taken at face value, so passing only ``Result.emotion``
            reads the ``("neutral", 0.0)`` of "no emotion reported" as a
            measured neutral speaker, which passes a rule that lists
            ``neutral`` as acceptable (it fails open).  It also sees a single
            emotion, not every emotion of the turn.  Gate on
            :meth:`evaluate_result` instead.
        context:
            Optional context dict.
        emotion_confidence:
            Confidence of *emotion*. Below *min_emotion_confidence* the
            emotion counts as unknown, which fails any required emotion
            list. Pass ``Result.confidence``: the ``("neutral", 0.0)`` that
            means "no emotion reported" is then not mistaken for a calm
            speaker. Prefer :meth:`evaluate_result`, which also weighs the
            other utterances of the turn.
        min_emotion_confidence:
            Confidence needed for *emotion* to count as evidence.

        Returns
        -------
        Decision
            Whether the action is allowed. Without a constitutional filter
            (no ``constitutional_rules``) every action is allowed.
        """
        if self._filter is None:
            return Decision(allow=True)

        return self._filter.evaluate(
            intent,
            prosody_features,
            emotion=emotion,
            context=context,
            emotion_confidence=emotion_confidence,
            min_emotion_confidence=min_emotion_confidence,
        )

    def evaluate_result(
        self,
        intent: str,
        result: Result,
        context: dict[str, object] | None = None,
    ) -> Decision:
        """Evaluate *intent* against the prosody and emotion in *result*.

        The usual way to gate an action: pass the intent the LLM parsed
        (``Response.intent``) and the :class:`Result` of the same turn.

        Every emotion the assembler reported with enough confidence counts,
        not only :attr:`Result.emotion` (the most confident utterance): the
        filter is run for that emotion and for each other utterance of
        ``result.iml_document`` that carries one, and the most restrictive
        decision wins.  A forbidden emotion in any one sentence of the turn
        therefore blocks the action, and a required emotion list has to be
        satisfied by every reported emotion.  Utterances without an emotion
        (the assembler abstained) are not evidence either way.

        When no emotion was reported at all, a single-utterance turn for
        example, the emotion is unknown, which fails a rule that requires a
        particular emotion (it needs verification rather than being allowed).
        ``Result.emotion`` and ``Result.confidence`` are used as they are when
        the result carries no usable document.

        Gating with :meth:`evaluate_intent` and only ``result.emotion`` sees
        one emotion per turn; use this method for anything that matters.
        """
        if self._filter is None:
            return Decision(allow=True)

        pairs: list[tuple[str | None, float | None]] = [(result.emotion, result.confidence)]
        document = getattr(result, "iml_document", None)
        for utterance in getattr(document, "utterances", None) or ():
            pair = (utterance.emotion, utterance.confidence)
            # An abstention, or an emotion below the threshold the filter
            # believes, says nothing; the primary pair is already in the list.
            if pair[1] is None or resolve_emotion(*pair) is None or pair in pairs:
                continue
            pairs.append(pair)

        return most_restrictive(
            self.evaluate_intent(
                intent,
                result.prosody_features,
                emotion=emotion,
                context=context,
                emotion_confidence=confidence,
            )
            for emotion, confidence in pairs
        )

    @staticmethod
    def _document_emotion(doc: IMLDocument) -> tuple[str, float]:
        """Emotion and confidence the assembler reported for *doc*.

        That is the utterance with the highest confidence among those
        that carry an emotion (the later one on a tie), or
        ``("neutral", 0.0)`` when the assembler abstained for all of them.
        """
        best: tuple[str, float] | None = None
        for utterance in doc.utterances:
            if utterance.emotion is None or utterance.confidence is None:
                continue
            if best is None or utterance.confidence >= best[1]:
                best = (utterance.emotion, utterance.confidence)
        return best if best is not None else ("neutral", 0.0)

    # -- Augmentative communication --

    async def type_to_speech(
        self, text: str, emotion: str = "neutral"
    ) -> Audio:
        """Convert typed text to emotionally appropriate speech.

        Supports augmentative and alternative communication (AAC) use
        cases where users type instead of speak.  The text is synthesized
        as typed, with the given emotion shaping the voice.  A TTS provider
        that reads SSML (it sets ``supports_ssml = True``) is instead given
        SSML built with ``prosody_protocol.TextToIML`` and ``IMLToSSML``,
        which predict pitch and pauses from the text; no built-in adapter
        does, and one that does not would speak the markup.

        Parameters
        ----------
        text:
            Plain text to convert to speech.
        emotion:
            Emotion label for synthesis.

        Returns
        -------
        Audio
            Synthesized speech audio.
        """
        if getattr(self._tts, "supports_ssml", False):
            from prosody_protocol import IMLToSSML, TextToIML

            iml_doc = TextToIML().predict(text, context=emotion)
            text = IMLToSSML().convert(iml_doc) or text

        return await self.synthesize_speech(text, emotion=emotion)

    def type_to_speech_sync(
        self, text: str, emotion: str = "neutral"
    ) -> Audio:
        """Synchronous wrapper for :meth:`type_to_speech`.

        Raises ``RuntimeError`` when called from a running event loop
        (async code, Jupyter, ...); ``await`` the async method there.
        """
        return self._run_sync(self.type_to_speech, text, emotion=emotion)

    # -- Profile management API --

    def load_profile(self, path: str | os.PathLike[str]) -> ProsodyProfile:
        """Load a prosody profile from a JSON file.

        Parameters
        ----------
        path:
            Path to a prosody profile JSON file conforming to
            ``schemas/prosody-profile.schema.json``.

        Returns
        -------
        ProsodyProfile
            The loaded profile.

        Raises
        ------
        ProfileError
            If the file cannot be read or parsed, or the profile does not
            validate (unknown pattern keys or values, a ``profile_version``
            that is not ``X.Y.Z``, no mappings, ...).
        """
        profile = self._profile_loader.load(Path(path))
        result = self._profile_loader.validate(profile)
        if not result.valid:
            problems = "; ".join(f"{i.rule}: {i.message}" for i in result.errors)
            raise ProfileError(f"Invalid prosody profile: {problems}")
        logger.info("Loaded prosody profile (%d mappings)", len(profile.mappings))
        return profile

    def set_profile(self, profile: ProsodyProfile) -> None:
        """Set the active prosody profile for this engine.

        The profile is applied by the IML assembler to each utterance,
        described in the profile vocabulary (``pitch``, ``pitch_contour``,
        ``volume``, ``rate``, ``quality``, ``pause_frequency``,
        ``emphasis_frequency``) against the speaker's baseline.  A matching
        mapping decides the utterance's emotion, which appears in the IML
        (marked ``x-profile``) and in ``Result``.

        Parameters
        ----------
        profile:
            A ``ProsodyProfile`` to apply in ``process_voice_input()``.

        Raises
        ------
        ProfileError
            If the profile does not validate (see :meth:`validate_profile`).
            The previously active profile stays in place.
        TypeError
            If *profile* is not a ``ProsodyProfile``.
        """
        # Building the assembler validates the profile, before any state changes
        assembler = IMLAssembler(
            emotion_classifier=self._emotion_classifier, profile=profile
        )
        with self._lock:
            self._assembler = assembler
            self._profile = profile
        logger.info("Active profile set (%d mappings)", len(profile.mappings))

    def clear_profile(self) -> None:
        """Remove the active prosody profile."""
        assembler = IMLAssembler(emotion_classifier=self._emotion_classifier)
        with self._lock:
            self._assembler = assembler
            self._profile = None
        logger.info("Active profile cleared")

    def create_profile(
        self,
        user_id: str,
        mappings: list[dict[str, Any]],
        description: str | None = None,
        profile_version: str = "1.0.0",
    ) -> ProsodyProfile:
        """Create a new prosody profile from mapping dicts.

        The profile is not validated here; use :meth:`validate_profile`
        (or :meth:`set_profile`, which rejects invalid profiles).

        Parameters
        ----------
        user_id:
            Unique identifier for the user.
        mappings:
            List of mapping dicts, each with ``"pattern"`` (dict),
            ``"interpretation_emotion"`` (str), and optional
            ``"confidence_boost"`` (float).
        description:
            Optional human-readable profile description.
        profile_version:
            Profile format version, as a semantic version (``X.Y.Z``).

        Returns
        -------
        ProsodyProfile
            The newly created (in-memory) profile.

        Raises
        ------
        ProfileError
            If a mapping dict lacks ``"pattern"`` or
            ``"interpretation_emotion"``.
        """
        pp_mappings = []
        for index, m in enumerate(mappings):
            try:
                pp_mappings.append(
                    ProsodyMapping(
                        pattern=m["pattern"],
                        interpretation_emotion=m["interpretation_emotion"],
                        confidence_boost=m.get("confidence_boost", 0.0),
                    )
                )
            except KeyError as exc:
                raise ProfileError(
                    f"mappings[{index}] is missing required key {exc}"
                ) from exc
        return ProsodyProfile(
            profile_version=profile_version,
            user_id=user_id,
            description=description,
            mappings=tuple(pp_mappings),
        )

    def validate_profile(self, profile: ProsodyProfile) -> ValidationResult:
        """Validate a prosody profile against the schema.

        Parameters
        ----------
        profile:
            The profile to validate.

        Returns
        -------
        ValidationResult
            Validation result from ``prosody_protocol``.
        """
        return self._profile_loader.validate(profile)

    # -- Synchronous convenience wrappers --

    def _run_sync(
        self,
        method: Callable[_P, Coroutine[Any, Any, _T]],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T:
        """Run an async method to completion from synchronous code.

        The calls of one engine share a background event loop (started on
        first use, see :meth:`close`), so async clients that provider SDKs
        cache keep working across calls.  Calling from a thread that is
        running an event loop would block that loop and is refused.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            name = method.__name__
            raise RuntimeError(
                f"{name}_sync() cannot be called from a running event loop; "
                f"use `await engine.{name}(...)` instead"
            )

        # Fetching the loop and scheduling on it happen under the lock that
        # close() takes to retire it: a call is either queued on the loop
        # before close() asks it to stop (and is cancelled with the rest) or
        # finds the new loop, never a loop that is already shutting down.
        with self._lock:
            runner = self._get_sync_runner_locked()
            coro = method(*args, **kwargs)
            try:
                future = asyncio.run_coroutine_threadsafe(coro, runner.loop)
            except BaseException:
                coro.close()  # never scheduled, so never awaited
                raise
        try:
            return future.result()
        except KeyboardInterrupt:
            future.cancel()
            raise
        finally:
            # The exception raised above references this frame; without this
            # the frame, the future and the exception form a cycle that keeps
            # the engine (and its loop thread) alive until the next GC pass.
            del future

    def _get_sync_runner(self) -> _LoopThread:
        with self._lock:
            return self._get_sync_runner_locked()

    def _get_sync_runner_locked(self) -> _LoopThread:
        runner = self._sync_runner
        # A forked child has the loop object but not its thread; a runner
        # whose thread has ended can serve no more calls
        if runner is None or runner.pid != os.getpid() or not runner.thread.is_alive():
            if runner is not None:
                runner.close()
            runner = self._sync_runner = _LoopThread(self)
        return runner

    def close(self) -> None:
        """Stop the background event loop used by the ``*_sync`` wrappers.

        The loop starts when a wrapper is first used and stops when the
        engine is garbage collected; call this to stop it sooner.  Safe to
        call more than once, and the wrappers start a new loop if used
        again.  A ``*_sync`` call still running in another thread is
        cancelled (its caller gets ``concurrent.futures.CancelledError``);
        one that starts while this runs is cancelled the same way or served
        by the new loop.
        """
        with self._lock:
            runner, self._sync_runner = self._sync_runner, None
        if runner is not None:
            runner.close()

    def process_voice_input_sync(
        self, audio_path: str, use_cache: bool = True
    ) -> Result:
        """Synchronous wrapper for :meth:`process_voice_input`.

        Raises ``RuntimeError`` when called from a running event loop
        (async code, Jupyter, ...); ``await`` the async method there.
        """
        return self._run_sync(
            self.process_voice_input, audio_path, use_cache=use_cache
        )

    def generate_response_sync(
        self,
        iml: str,
        context: str | None = None,
        tone: str | None = None,
    ) -> Response:
        """Synchronous wrapper for :meth:`generate_response`.

        Raises ``RuntimeError`` when called from a running event loop
        (async code, Jupyter, ...); ``await`` the async method there.
        """
        return self._run_sync(
            self.generate_response, iml, context=context, tone=tone
        )

    def synthesize_speech_sync(
        self, text: str, emotion: str = "neutral"
    ) -> Audio:
        """Synchronous wrapper for :meth:`synthesize_speech`.

        Raises ``RuntimeError`` when called from a running event loop
        (async code, Jupyter, ...); ``await`` the async method there.
        """
        return self._run_sync(self.synthesize_speech, text, emotion=emotion)
