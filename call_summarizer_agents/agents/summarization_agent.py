"""Summarization agent that transforms transcripts into concise outputs."""
from __future__ import annotations

from textwrap import shorten
from typing import Any, Dict, Iterable, Optional

from openai import OpenAI

from call_summarizer_agents.utils.validation import SummaryPayload
try:
    from langsmith.wrappers import wrap_openai
except Exception:
    wrap_openai = None

class SummarizationAgent:
    """Generate summaries and key insights from transcripts."""

    def __init__(
        self,
        llm: Any | None = None,
        openai_api_key: str | None = None,
        openai_model: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ) -> None:
        self.name = "SummarizationAgent"
        self.llm = llm
        self.openai_model = openai_model
        self.temperature = temperature
        client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        if client and wrap_openai:
            client = wrap_openai(client)
        self._openai_client: Optional[OpenAI] = (
            client
        )

    def __call__(self, payload: Dict[str, Any]) -> SummaryPayload:
        transcript: str = payload["transcript"]
        conversation_id: str = payload.get("conversation_id", "unknown")

        if self.llm:
            summary = self._run_llm(transcript)
        elif self._openai_client:
            summary = self._run_openai(transcript)
        else:
            summary = self._fallback_summary(transcript)
        key_points = self._extract_key_points(transcript)
        risks = self._extract_risks(transcript)
        follow_ups = self._extract_followups(transcript)

        return SummaryPayload(
            conversation_id=conversation_id,
            summary=summary,
            key_points=key_points,
            risks=risks,
            follow_ups=follow_ups,
        )

    def _run_llm(self, transcript: str) -> str:
        """Invoke an LLM via LangChain, if provided."""

        prompt = (
            "Summarize the following customer support call in four bullet points. "
            "Be concise and avoid speculation.\n\n" + transcript
        )
        response = self.llm.invoke(prompt)
        return getattr(response, "content", str(response))

    def _run_openai(self, transcript: str) -> str:
        """Invoke OpenAI chat completions when an API key is provided."""

        try:
            response = self._openai_client.chat.completions.create(
                model=self.openai_model,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You summarize contact center calls and return concise, factual bullets."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Summarize the following customer support call in four bullet points. "
                            "Be concise and avoid speculation.\n\n" + transcript
                        ),
                    },
                ],
            )
        except Exception:
            return self._fallback_summary(transcript)

        content = response.choices[0].message.content if response.choices else None
        return content or self._fallback_summary(transcript)

    def _fallback_summary(self, transcript: str) -> str:
        """Deterministic summary used for local testing without an API key."""

        short = shorten(transcript.replace("\n", " "), width=400, placeholder="...")
        return f"Auto-generated summary (rule-based): {short}"

    def _extract_key_points(self, transcript: str) -> list[str]:
        sentences = [s.strip() for s in transcript.split(".") if s.strip()]
        return sentences[:4]

    def _extract_risks(self, transcript: str) -> list[str]:
        risk_keywords = ("cancel", "refund", "angry", "escalate", "complaint")
        return _find_sentences_with_keywords(transcript, risk_keywords)

    def _extract_followups(self, transcript: str) -> list[str]:
        followup_keywords = ("follow", "email", "case", "ticket", "tomorrow", "next week")
        return _find_sentences_with_keywords(transcript, followup_keywords)


def _find_sentences_with_keywords(transcript: str, keywords: Iterable[str]) -> list[str]:
    sentences = [s.strip() for s in transcript.split(".") if s.strip()]
    matches: list[str] = []
    for sentence in sentences:
        if any(keyword.lower() in sentence.lower() for keyword in keywords):
            matches.append(sentence)
    return matches
