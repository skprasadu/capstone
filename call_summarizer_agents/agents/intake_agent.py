"""Call intake agent that validates inputs and extracts metadata."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from call_summarizer_agents.utils.validation import CallInput


class CallIntakeAgent:
    """Validate call payloads and produce normalized metadata."""

    def __init__(self) -> None:
        self.name = "CallIntakeAgent"

    def __call__(self, raw_payload: Dict[str, Any]) -> CallInput:
        """Validate inbound payload and fill defaults.

        Args:
            raw_payload: Arbitrary payload from API or UI.

        Returns:
            CallInput: Parsed and validated payload.
        """

        payload = CallInput(**raw_payload)
        return payload

    def extract_metadata(self, payload: CallInput) -> dict[str, Any]:
        """Build lightweight metadata for logging and observability."""

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
