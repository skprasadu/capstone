"""Call intake agent that validates inputs and extracts metadata."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
import json
import re

from openai import OpenAI

from call_summarizer_agents.utils.validation import CallInput

from capstone_common.llm.openai_client import get_openai_client

# Matches: Karla: "Hello..."
_SPEAKER_LINE = re.compile(r'^\s*([A-Za-z][A-Za-z0-9_.\- ]{0,40})\s*:\s*(.*)\s*$')

# conservative name cleanup
_BAD_NAMES = {"agent", "customer", "caller", "client", "representative", "support", "csr", "unknown", "n/a"}


_PARTICIPANTS_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_participants",
        "description": (
            "Extract the agent name and customer name from a support call transcript. "
            "Return null for any name not explicitly present in the text. Do not guess."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "agent_name": {"type": ["string", "null"]},
                "agent_evidence": {
                    "type": ["string", "null"],
                    "description": "A short exact snippet from the transcript that supports agent_name.",
                },
                "customer_name": {"type": ["string", "null"]},
                "customer_evidence": {
                    "type": ["string", "null"],
                    "description": "A short exact snippet from the transcript that supports customer_name.",
                },
            },
            "required": ["agent_name", "agent_evidence", "customer_name", "customer_evidence"],
        },
    },
}


def _strip_quotes(s: str) -> str:
    return s.strip().strip('"\u201c\u201d\u2018\u2019').strip()


def _normalize_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = _strip_quotes(name)
    n = re.sub(r"\s+", " ", n).strip()
    if not n:
        return None
    if n.lower() in _BAD_NAMES:
        return None
    # avoid absurd outputs
    if len(n) < 2 or len(n) > 64:
        return None
    return n


def _has_speaker_tags(transcript: str) -> bool:
    speakers: set[str] = set()
    hits = 0
    for line in (transcript or "").splitlines():
        m = _SPEAKER_LINE.match(line.strip())
        if not m:
            continue
        hits += 1
        speakers.add(_strip_quotes(m.group(1)).lower())
        if hits >= 2 and len(speakers) >= 2:
            return True
    return False


def _infer_from_speaker_tags(transcript: str) -> tuple[Optional[str], Optional[str]]:
    turns: list[tuple[str, str]] = []
    for line in (transcript or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = _SPEAKER_LINE.match(line)
        if not m:
            continue
        speaker = _strip_quotes(m.group(1))
        text = _strip_quotes(m.group(2))
        if speaker and text:
            turns.append((speaker, text))

    speakers: list[str] = []
    for s, _ in turns:
        if s not in speakers:
            speakers.append(s)

    if len(speakers) < 2:
        return None, None

    # Heuristic: likely agent is the one with greeting/courtesy phrases
    def agent_score(text: str) -> int:
        t = text.lower()
        score = 0
        rules = [
            ("thank you for calling", 3),
            ("thanks for calling", 3),
            ("how can i help", 2),
            ("how may i help", 2),
            ("this is", 1),
            ("my name is", 1),
            ("goodbye", 1),
            ("have a wonderful day", 1),
            ("have a good day", 1),
        ]
        for pat, w in rules:
            if pat in t:
                score += w
        return score

    scores = {s: 0 for s in speakers}
    for s, text in turns[:10]:
        scores[s] += agent_score(text)

    agent = max(speakers, key=lambda s: (scores[s], -speakers.index(s)))
    customer = next((s for s in speakers if s != agent), None)
    return _normalize_name(agent), _normalize_name(customer)


class CallIntakeAgent:
    """Validate call payloads and produce normalized metadata."""

    def __init__(
        self,
        openai_api_key: str | None = None,
        openai_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        client: Any | None = None,
    ) -> None:
        self.name = "CallIntakeAgent"
        self.openai_model = openai_model
        self.temperature = temperature

        self._openai_client: Any | None = get_openai_client(
            openai_api_key,
            client=client,
            wrap_langsmith=True,
        )

    def __call__(self, raw_payload: Dict[str, Any]) -> CallInput:
        payload = CallInput(**raw_payload)

        # If transcript already provided (chat), fill missing participants now
        if payload.transcript and (payload.agent_name is None or payload.customer_name is None):
            a, c = self.infer_participants(
                payload.transcript,
                agent_name=payload.agent_name,
                customer_name=payload.customer_name,
            )
            if payload.agent_name is None and a:
                payload.agent_name = a
            if payload.customer_name is None and c:
                payload.customer_name = c

        return payload

    def infer_participants(
        self,
        transcript: str,
        agent_name: Optional[str] = None,
        customer_name: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        # Don’t override explicit values
        agent_name = _normalize_name(agent_name)
        customer_name = _normalize_name(customer_name)
        if agent_name and customer_name:
            return agent_name, customer_name

        # If transcript is clearly speaker-tagged, regex is actually the most precise.
        if transcript and _has_speaker_tags(transcript):
            a, c = _infer_from_speaker_tags(transcript)
            return agent_name or a, customer_name or c

        # For WAV->Whisper plain text, use LLM tool-calling (precision-first)
        if self._openai_client and transcript and transcript.strip():
            a, c = self._infer_with_llm(transcript)
            return agent_name or a, customer_name or c

        return agent_name, customer_name

    def _infer_with_llm(self, transcript: str) -> tuple[Optional[str], Optional[str]]:
        try:
            resp = self._openai_client.chat.completions.create(
                model=self.openai_model,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract participant names from the transcript.\n"
                            "- Return null if the name is NOT explicitly present.\n"
                            "- Do NOT guess.\n"
                            "- Provide evidence snippets copied exactly from the transcript.\n"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Transcript:\n{transcript}\n\nReturn ONLY via the tool.",
                    },
                ],
                tools=[_PARTICIPANTS_TOOL],
                tool_choice={"type": "function", "function": {"name": "emit_participants"}},
            )

            msg = resp.choices[0].message if resp.choices else None
            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                return None, None

            args = tool_calls[0].function.arguments
            data = json.loads(args) if isinstance(args, str) else (args or {})
            if not isinstance(data, dict):
                return None, None

            agent = _normalize_name(data.get("agent_name"))
            customer = _normalize_name(data.get("customer_name"))
            agent_ev = (data.get("agent_evidence") or "").strip()
            cust_ev = (data.get("customer_evidence") or "").strip()

            t_low = transcript.lower()

            def ok(name: Optional[str], ev: str) -> bool:
                if not name:
                    return False
                if not ev:
                    return False
                if ev.lower() not in t_low:
                    return False
                # also require the name itself appears (word-ish match)
                return re.search(rf"\b{re.escape(name)}\b", transcript, flags=re.IGNORECASE) is not None

            if agent and not ok(agent, agent_ev):
                agent = None
            if customer and not ok(customer, cust_ev):
                customer = None

            return agent, customer

        except Exception:
            return None, None

    def extract_metadata(self, payload: CallInput) -> dict[str, Any]:
        started_at = datetime.utcnow().isoformat()
        return {
            "conversation_id": payload.conversation_id,
            "agent_name": payload.agent_name,
            "customer_name": payload.customer_name,
            "channel": payload.channel,
            "ingested_at": started_at,
            "has_audio": bool(payload.audio_path),
            "has_transcript": bool(payload.transcript),
        }