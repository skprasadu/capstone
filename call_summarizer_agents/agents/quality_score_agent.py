"""Quality scoring agent with a structured rubric."""
from __future__ import annotations

from typing import Any, Dict

from call_summarizer_agents.utils.validation import QualityScore


class QualityScoreAgent:
    """Score conversations for professionalism, empathy, resolution, and compliance."""

    def __init__(self, rubric: Dict[str, str] | None = None) -> None:
        self.name = "QualityScoreAgent"
        self.rubric = rubric or {
            "professionalism": "Tone, greeting, and courtesy standards",
            "empathy": "Understood and acknowledged customer sentiment",
            "resolution": "Clear path to resolve the issue",
            "compliance": "Adhered to disclaimers and verification",
        }

    def __call__(self, payload: Dict[str, Any]) -> QualityScore:
        transcript: str = payload["transcript"]
        conversation_id: str = payload.get("conversation_id", "unknown")

        professionalism = self._score_presence(transcript, ["thank", "appreciate", "help"])
        empathy = self._score_presence(transcript, ["sorry", "understand", "apologize"])
        resolution = self._score_presence(transcript, ["resolved", "solution", "fixed", "sent"])  # noqa: E501
        compliance = self._score_presence(transcript, ["policy", "verify", "recorded", "consent"])

        feedback = (
            "Pseudo scores based on keyword coverage. Integrate function-calling LLMs "
            "for production-quality QA."
        )

        risks = self._collect_risks(transcript)

        return QualityScore(
            conversation_id=conversation_id,
            professionalism=professionalism,
            empathy=empathy,
            resolution=resolution,
            compliance=compliance,
            overall=round((professionalism + empathy + resolution + compliance) / 4),
            summary_feedback=feedback,
            risks=risks,
        )

    def _score_presence(self, transcript: str, keywords: list[str]) -> int:
        matches = sum(1 for keyword in keywords if keyword.lower() in transcript.lower())
        return min(5, max(1, matches))

    def _collect_risks(self, transcript: str) -> list[str]:
        risk_phrases = ["cancel", "escalate", "sue", "violation", "refund"]
        return [phrase for phrase in risk_phrases if phrase in transcript.lower()]
