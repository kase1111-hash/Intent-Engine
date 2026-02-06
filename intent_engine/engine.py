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
import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

from prosody_protocol import (
    IMLAssembler,
    IMLParser,
    IMLValidator,
    ProfileApplier,
    ProfileLoader,
    ProsodyAnalyzer,
    ProsodyMapping,
    ProsodyProfile,
    RuleBasedEmotionClassifier,
    SpanFeatures,
    ValidationResult,
)

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
        Optional path to a YAML file with constitutional rules.
    prosody_profile:
        Optional path to a prosody profile JSON for atypical
        prosody handling.
    cache_size:
        Maximum number of audio results to cache (LRU).
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
        constitutional_rules: str | None = None,
        prosody_profile: str | None = None,
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

        # Prosody Protocol components
        self._analyzer = ProsodyAnalyzer()
        self._assembler = IMLAssembler()
        self._parser = IMLParser()
        self._validator = IMLValidator()
        self._emotion_classifier = RuleBasedEmotionClassifier()

        # Optional constitutional filter
        self._filter: ConstitutionalFilter | None = None
        if constitutional_rules:
            self._filter = ConstitutionalFilter.from_yaml(constitutional_rules)

        # Profile components (always available for management API)
        self._profile_loader = ProfileLoader()
        self._profile_applier = ProfileApplier()

        # Optional accessibility profile (loaded from path)
        self._profile: ProsodyProfile | None = None
        if prosody_profile:
            self._profile = self._profile_loader.load(prosody_profile)

        # LRU cache for audio processing results
        self._cache_size = cache_size
        self._cache: OrderedDict[str, Result] = OrderedDict()

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
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, result: Result) -> None:
        """Store a result in the cache, evicting the oldest if full."""
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = result

    @staticmethod
    def _audio_hash(audio_path: str) -> str:
        """Compute a hash of an audio file for cache keying."""
        h = hashlib.sha256()
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

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
            7. EmotionClassifier classifies emotion from features
            8. Return Result with all data

        Parameters
        ----------
        audio_path:
            Path to the audio file.
        use_cache:
            Whether to use cached results for the same audio file.

        Returns
        -------
        Result
            Full pipeline result including IML, features, and emotion.

        Raises
        ------
        STTError
            If transcription fails.
        IntentEngineError
            If IML validation fails.
        """
        # Validate audio path
        audio = Path(audio_path)
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not audio.is_file():
            raise ValueError(f"Audio path is not a file: {audio_path}")

        # Check cache
        if use_cache:
            cache_key = self._audio_hash(audio_path)
            cached = self._cache_get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for %s", audio_path)
                return cached

        # Step 1: STT transcription
        try:
            transcription = await self._stt.transcribe(audio_path)
        except Exception as exc:
            raise STTError(f"STT transcription failed: {exc}") from exc

        # Steps 2-3: Prosody analysis (with fallback)
        features: list[SpanFeatures] = []
        pauses = []
        try:
            features = self._analyzer.analyze(
                audio_path, transcription.alignments
            )
            pauses = self._analyzer.detect_pauses(audio_path)
        except (RuntimeError, OSError, ValueError):
            logger.warning(
                "Prosody analysis failed for '%s'; falling back to text-only IML",
                audio_path,
                exc_info=True,
            )

        # Step 4: IML assembly
        iml_doc = self._assembler.assemble(
            transcription.alignments,
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

        # Step 7: Emotion classification
        emotion, confidence = self._emotion_classifier.classify(features)

        # Apply prosody profile if available
        if self._profile and features:
            try:
                feat_dict = self._derive_feature_labels(features)
                emotion, confidence = self._profile_applier.apply(
                    self._profile, feat_dict, emotion, confidence
                )
            except (RuntimeError, KeyError, ValueError):
                logger.warning(
                    "Prosody profile application failed; using base classification",
                    exc_info=True,
                )

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

        # Cache result
        if use_cache:
            self._cache_put(cache_key, result)  # type: ignore[possibly-undefined]

        return result

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
            Suggested tone to pass as context hint.

        Returns
        -------
        Response
            Response text and emotion for TTS.

        Raises
        ------
        LLMError
            If the LLM call fails.
        """
        full_context = context or ""
        if tone:
            tone_hint = f"Respond with a '{tone}' tone."
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
    ) -> Decision:
        """Evaluate an intent through the constitutional filter.

        Parameters
        ----------
        intent:
            Parsed user intent label.
        prosody_features:
            Prosody features from the pipeline.
        emotion:
            Detected emotion label.
        context:
            Optional context dict.

        Returns
        -------
        Decision
            Whether the action is allowed.
        """
        if self._filter is None:
            return Decision(allow=True)

        return self._filter.evaluate(
            intent, prosody_features, emotion=emotion, context=context
        )

    # -- Feature label derivation --

    # Thresholds for converting numeric SpanFeatures to categorical labels.
    # These baselines approximate typical adult speech.
    _F0_LOW = 120.0   # Hz -- below this is "low" pitch
    _F0_HIGH = 220.0  # Hz -- above this is "high" pitch
    _INTENSITY_QUIET = 55.0   # dB
    _INTENSITY_LOUD = 75.0    # dB
    _RATE_SLOW = 3.0   # syllables/sec
    _RATE_FAST = 6.0   # syllables/sec

    @classmethod
    def _derive_feature_labels(
        cls, features: list[SpanFeatures]
    ) -> dict[str, str]:
        """Convert numeric SpanFeatures into categorical labels.

        The profile system uses categorical labels (``"high"``,
        ``"low"``, ``"normal"``) rather than raw numbers.  This
        method aggregates across all spans and produces a label
        dict suitable for ``ProfileApplier.apply()``.

        Parameters
        ----------
        features:
            List of per-span prosodic features from
            ``ProsodyAnalyzer.analyze()``.

        Returns
        -------
        dict[str, str]
            Categorical labels for profile matching, e.g.
            ``{"f0_mean": "high", "intensity_mean": "normal",
            "speech_rate": "fast", "quality": "breathy"}``.
        """
        labels: dict[str, str] = {}

        if not features:
            return labels

        # Aggregate numeric values across spans (use first span as primary)
        f0_values = [f.f0_mean for f in features if f.f0_mean is not None]
        intensity_values = [
            f.intensity_mean for f in features if f.intensity_mean is not None
        ]
        rate_values = [
            f.speech_rate for f in features if f.speech_rate is not None
        ]

        # F0 (pitch)
        if f0_values:
            avg_f0 = sum(f0_values) / len(f0_values)
            if avg_f0 < cls._F0_LOW:
                labels["f0_mean"] = "low"
            elif avg_f0 > cls._F0_HIGH:
                labels["f0_mean"] = "high"
            else:
                labels["f0_mean"] = "normal"

        # Intensity (loudness)
        if intensity_values:
            avg_intensity = sum(intensity_values) / len(intensity_values)
            if avg_intensity < cls._INTENSITY_QUIET:
                labels["intensity_mean"] = "quiet"
            elif avg_intensity > cls._INTENSITY_LOUD:
                labels["intensity_mean"] = "loud"
            else:
                labels["intensity_mean"] = "normal"

        # Speech rate
        if rate_values:
            avg_rate = sum(rate_values) / len(rate_values)
            if avg_rate < cls._RATE_SLOW:
                labels["speech_rate"] = "slow"
            elif avg_rate > cls._RATE_FAST:
                labels["speech_rate"] = "fast"
            else:
                labels["speech_rate"] = "normal"

        # Voice quality (pass through directly if present)
        qualities = [f.quality for f in features if f.quality is not None]
        if qualities:
            labels["quality"] = qualities[0]

        return labels

    # -- Augmentative communication --

    async def type_to_speech(
        self, text: str, emotion: str = "neutral"
    ) -> Audio:
        """Convert typed text to emotionally appropriate speech.

        Uses ``prosody_protocol.TextToIML`` to predict prosody from
        plain text, then synthesizes with the given emotion.  This
        supports augmentative and alternative communication (AAC)
        use cases where users type instead of speak.

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
        from prosody_protocol import IMLToSSML, TextToIML

        predictor = TextToIML()
        iml_doc = predictor.predict(text, context=emotion)

        # Convert the predicted IML to SSML for richer synthesis if possible
        ssml_converter = IMLToSSML()
        ssml_text = ssml_converter.convert(iml_doc)

        return await self.synthesize_speech(ssml_text or text, emotion=emotion)

    def type_to_speech_sync(
        self, text: str, emotion: str = "neutral"
    ) -> Audio:
        """Synchronous wrapper for :meth:`type_to_speech`."""
        return asyncio.run(self.type_to_speech(text, emotion=emotion))

    # -- Profile management API --

    def load_profile(self, path: str) -> ProsodyProfile:
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
        """
        profile = self._profile_loader.load(path)
        logger.info("Loaded prosody profile: %s", profile.user_id)
        return profile

    def set_profile(self, profile: ProsodyProfile) -> None:
        """Set the active prosody profile for this engine.

        Parameters
        ----------
        profile:
            A ``ProsodyProfile`` to apply during emotion
            classification in ``process_voice_input()``.
        """
        self._profile = profile
        logger.info("Active profile set: %s", profile.user_id)

    def clear_profile(self) -> None:
        """Remove the active prosody profile."""
        self._profile = None
        logger.info("Active profile cleared")

    def create_profile(
        self,
        user_id: str,
        mappings: list[dict[str, Any]],
        description: str | None = None,
        profile_version: str = "1.0",
    ) -> ProsodyProfile:
        """Create a new prosody profile from mapping dicts.

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
            Profile schema version.

        Returns
        -------
        ProsodyProfile
            The newly created (in-memory) profile.
        """
        pp_mappings = [
            ProsodyMapping(
                pattern=m["pattern"],
                interpretation_emotion=m["interpretation_emotion"],
                confidence_boost=m.get("confidence_boost", 0.0),
            )
            for m in mappings
        ]
        return ProsodyProfile(
            profile_version=profile_version,
            user_id=user_id,
            description=description,
            mappings=pp_mappings,
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

    def process_voice_input_sync(
        self, audio_path: str, use_cache: bool = True
    ) -> Result:
        """Synchronous wrapper for :meth:`process_voice_input`."""
        return asyncio.run(
            self.process_voice_input(audio_path, use_cache=use_cache)
        )

    def generate_response_sync(
        self,
        iml: str,
        context: str | None = None,
        tone: str | None = None,
    ) -> Response:
        """Synchronous wrapper for :meth:`generate_response`."""
        return asyncio.run(
            self.generate_response(iml, context=context, tone=tone)
        )

    def synthesize_speech_sync(
        self, text: str, emotion: str = "neutral"
    ) -> Audio:
        """Synchronous wrapper for :meth:`synthesize_speech`."""
        return asyncio.run(self.synthesize_speech(text, emotion=emotion))
