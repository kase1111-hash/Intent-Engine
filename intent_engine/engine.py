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
from pathlib import Path
from typing import Any

from prosody_protocol import (
    IMLAssembler,
    IMLParser,
    IMLValidator,
    ProsodyAnalyzer,
    RuleBasedEmotionClassifier,
    SpanFeatures,
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

        # Optional accessibility profile
        self._profile: object | None = None
        self._profile_applier: object | None = None
        if prosody_profile:
            from prosody_protocol import ProfileApplier, ProfileLoader

            self._profile = ProfileLoader().load(prosody_profile)
            self._profile_applier = ProfileApplier()

        # LRU cache for audio processing results
        self._cache_size = cache_size
        self._cache: dict[str, Result] = {}
        self._cache_order: list[str] = []

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
            # Move to end (most recently used)
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, result: Result) -> None:
        """Store a result in the cache, evicting the oldest if full."""
        if key in self._cache:
            self._cache_order.remove(key)
        elif len(self._cache) >= self._cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        self._cache[key] = result
        self._cache_order.append(key)

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
        except Exception:
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
        if self._profile and self._profile_applier and features:
            try:
                feat_dict = {}
                if features[0].f0_mean is not None:
                    feat_dict["f0_mean"] = str(features[0].f0_mean)
                if features[0].speech_rate is not None:
                    feat_dict["speech_rate"] = str(features[0].speech_rate)
                emotion, confidence = self._profile_applier.apply(  # type: ignore[union-attr]
                    self._profile, feat_dict, emotion, confidence
                )
            except Exception:
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

    # -- Synchronous convenience wrappers --

    def process_voice_input_sync(
        self, audio_path: str, use_cache: bool = True
    ) -> Result:
        """Synchronous wrapper for :meth:`process_voice_input`."""
        return asyncio.get_event_loop().run_until_complete(
            self.process_voice_input(audio_path, use_cache=use_cache)
        )

    def generate_response_sync(
        self,
        iml: str,
        context: str | None = None,
        tone: str | None = None,
    ) -> Response:
        """Synchronous wrapper for :meth:`generate_response`."""
        return asyncio.get_event_loop().run_until_complete(
            self.generate_response(iml, context=context, tone=tone)
        )

    def synthesize_speech_sync(
        self, text: str, emotion: str = "neutral"
    ) -> Audio:
        """Synchronous wrapper for :meth:`synthesize_speech`."""
        return asyncio.get_event_loop().run_until_complete(
            self.synthesize_speech(text, emotion=emotion)
        )
