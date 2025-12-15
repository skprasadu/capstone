"""Routing agent orchestrating the multi-agent workflow."""
from __future__ import annotations

from typing import Any, Dict, Optional

from call_summarizer_agents.agents.intake_agent import CallIntakeAgent
from call_summarizer_agents.agents.quality_score_agent import QualityScoreAgent
from call_summarizer_agents.agents.summarization_agent import SummarizationAgent
from call_summarizer_agents.agents.transcription_agent import TranscriptionAgent
from call_summarizer_agents.config.settings import AppSettings, load_settings
from call_summarizer_agents.utils.validation import CallInput, QualityScore, SummaryPayload, TranscriptPayload


class RoutingAgent:
    """Coordinate agent execution, handle fallbacks, and capture observability data."""

    def __init__(
        self,
        intake_agent: Optional[CallIntakeAgent] = None,
        transcription_agent: Optional[TranscriptionAgent] = None,
        summarization_agent: Optional[SummarizationAgent] = None,
        quality_agent: Optional[QualityScoreAgent] = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.intake_agent = intake_agent or CallIntakeAgent()
        whisper_key = self.settings.whisper_api_key or self.settings.openai_api_key
        self.transcription_agent = TranscriptionAgent(
            whisper_api_key=whisper_key,
            whisper_model=self.settings.whisper_model,
        )
        self.summarization_agent = summarization_agent or SummarizationAgent(
            openai_api_key=self.settings.openai_api_key,
            openai_model=self.settings.openai_model,
            temperature=self.settings.openai_temperature,
        )
        self.quality_agent = quality_agent or QualityScoreAgent(
            openai_api_key=self.settings.openai_api_key,
            openai_model=self.settings.openai_model,
            temperature=0.0,
        )

    def run(self, payload: Dict[str, Any]) -> dict[str, Any]:
        """Execute the multi-step pipeline with fallbacks."""

        intake = self._ingest(payload)
        transcript_payload = self._transcribe(intake)
        summary_payload = self._summarize(transcript_payload)
        quality_payload = self._quality_check(transcript_payload, summary_payload)

        return {
            "metadata": self.intake_agent.extract_metadata(intake),
            "transcript": transcript_payload.model_dump(),
            "summary": summary_payload.model_dump(),
            "quality": quality_payload.model_dump(),
        }

    def _ingest(self, payload: Dict[str, Any]) -> CallInput:
        return self.intake_agent(payload)

    def _transcribe(self, payload: CallInput) -> TranscriptPayload:
        return self.transcription_agent(payload.model_dump())

    def _summarize(self, transcript: TranscriptPayload) -> SummaryPayload:
        return self.summarization_agent(transcript.model_dump())

    def _quality_check(
        self, transcript: TranscriptPayload, summary: SummaryPayload
    ) -> QualityScore:
        merged_payload: dict[str, Any] = {
            **transcript.model_dump(),
            **summary.model_dump(),
        }
        return self.quality_agent(merged_payload)
