"""Quality scoring agent with a structured rubric (LLM tool-calling + heuristic fallback)."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from openai import OpenAI

from call_summarizer_agents.utils.validation import QualityScore

from capstone_common.llm.openai_client import get_openai_client

_QA_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_quality_score",
        "description": (
            "Return QA rubric scores (1-5) for a customer support interaction. "
            "Also return short feedback and a list of risks detected."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "professionalism": {"type": "integer", "minimum": 1, "maximum": 5},
                "empathy": {"type": "integer", "minimum": 1, "maximum": 5},
                "resolution": {"type": "integer", "minimum": 1, "maximum": 5},
                "compliance": {"type": "integer", "minimum": 1, "maximum": 5},
                "summary_feedback": {
                    "type": "string",
                    "description": "1-3 sentences. Mention brief evidence from the transcript."
                },
                "risks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Short phrases like 'refund request', 'escalation threat', 'compliance gap'."
                },
            },
            "required": ["professionalism", "empathy", "resolution", "compliance", "summary_feedback", "risks"],
        },
    },
}


class QualityScoreAgent:
    """Score conversations for professionalism, empathy, resolution, and compliance."""

    def __init__(
        self,
        rubric: Dict[str, str] | None = None,
        openai_api_key: str | None = None,
        openai_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        client: Any | None = None,
    ) -> None:
        self.name = "QualityScoreAgent"
        self.rubric = rubric or {
            "professionalism": "Tone, greeting, and courtesy standards",
            "empathy": "Understood and acknowledged customer sentiment",
            "resolution": "Clear path to resolve the issue",
            "compliance": "Adhered to disclaimers and verification",
        }
        self.openai_model = openai_model
        self.temperature = temperature

        # Optional OpenAI client for tool-calling QA
        self._openai_client: Optional[Any] = get_openai_client(
            openai_api_key,
            client=client,
            wrap_langsmith=True,
        )
    def __call__(self, payload: Dict[str, Any]) -> QualityScore:
        # Prefer LLM tool-calling if configured; fallback to heuristic always exists.
        llm_score = self._score_with_llm(payload) if self._openai_client else None
        if llm_score is not None:
            return llm_score

        return self._score_heuristic(payload)

    # -------------------------
    # LLM scoring (tool-calling)
    # -------------------------

    def _score_with_llm(self, payload: Dict[str, Any]) -> QualityScore | None:
        transcript: str = payload.get("transcript") or ""
        if not transcript.strip():
            return None

        conversation_id: str = payload.get("conversation_id", "unknown")
        summary_text: str = payload.get("summary") or ""
        key_points: list[str] = payload.get("key_points") or []

        rubric_text = "\n".join(f"- {k}: {v}" for k, v in self.rubric.items())

        user_content = (
            "Score this call using the rubric. Return ONLY via the tool.\n\n"
            f"Rubric:\n{rubric_text}\n\n"
            f"Transcript:\n{transcript}\n"
        )
        if summary_text:
            user_content += f"\nExisting summary (optional context):\n{summary_text}\n"
        if key_points:
            user_content += "\nKey points (optional context):\n" + "\n".join(f"- {kp}" for kp in key_points) + "\n"

        try:
            resp = self._openai_client.chat.completions.create(
                model=self.openai_model,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict QA auditor for contact center calls. "
                            "Use ONLY the provided transcript/context. "
                            "Scores are integers 1-5. If unclear, choose 3 (neutral). "
                            "Keep feedback short and cite brief evidence."
                        ),
                    },
                    {"role": "user", "content": user_content},
                ],
                tools=[_QA_TOOL],
                tool_choice={"type": "function", "function": {"name": "emit_quality_score"}},
            )

            msg = resp.choices[0].message if resp.choices else None
            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                return None

            args = tool_calls[0].function.arguments
            data = json.loads(args) if isinstance(args, str) else (args or {})
            if not isinstance(data, dict):
                return None

            # Validate required scores
            scores = [
                data.get("professionalism"),
                data.get("empathy"),
                data.get("resolution"),
                data.get("compliance"),
            ]
            if any(not isinstance(s, int) for s in scores):
                return None

            # Clamp and compute overall ourselves (don’t trust model math)
            def clamp(x: int) -> int:
                return max(1, min(5, int(x)))

            professionalism = clamp(data["professionalism"])
            empathy = clamp(data["empathy"])
            resolution = clamp(data["resolution"])
            compliance = clamp(data["compliance"])
            overall = clamp(round((professionalism + empathy + resolution + compliance) / 4))

            summary_feedback = str(data.get("summary_feedback") or "").strip()
            if not summary_feedback:
                summary_feedback = "LLM QA scoring completed."

            risks = data.get("risks")
            if not isinstance(risks, list):
                risks = []
            risks = [str(r).strip() for r in risks if str(r).strip()]

            return QualityScore(
                conversation_id=conversation_id,
                professionalism=professionalism,
                empathy=empathy,
                resolution=resolution,
                compliance=compliance,
                overall=overall,
                summary_feedback=summary_feedback,
                risks=risks,
            )

        except Exception:
            # Any parse/API issues -> fallback
            return None

    # -------------------------
    # Heuristic fallback (existing behavior)
    # -------------------------

    def _score_heuristic(self, payload: Dict[str, Any]) -> QualityScore:
        transcript: str = payload["transcript"]
        conversation_id: str = payload.get("conversation_id", "unknown")

        professionalism = self._score_presence(transcript, ["thank", "appreciate", "help"])
        empathy = self._score_presence(transcript, ["sorry", "understand", "apologize"])
        resolution = self._score_presence(transcript, ["resolved", "solution", "fixed", "sent"])
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