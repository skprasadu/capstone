"""Validation helpers and data contracts for call summarization agents."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class CallInput(BaseModel):
    """Validated input for the call pipeline."""

    conversation_id: str = Field(..., description="Unique identifier for the call session")
    agent_name: str = Field(..., description="Contact center agent who handled the call")
    customer_name: str = Field(..., description="Customer involved in the interaction")
    audio_path: Optional[Path] = Field(
        None, description="Optional local path to the recorded call audio file"
    )
    transcript: Optional[str] = Field(
        None, description="Pre-existing transcript text, if already available"
    )
    channel: str = Field(
        "voice", description="Interaction channel such as voice, chat, or email"
    )

    @model_validator(mode="after")
    def ensure_content(self) -> "CallInput":
        if not self.audio_path and not self.transcript:
            msg = "Either an audio_path or transcript must be supplied."
            raise ValueError(msg)
        return self


class TranscriptPayload(BaseModel):
    """Structured transcript content shared across downstream agents."""

    conversation_id: str
    transcript: str
    audio_path: Optional[Path] = None
    duration_seconds: Optional[float] = Field(
        None, description="Duration of the call if known from metadata"
    )


class SummaryPayload(BaseModel):
    """Structured summary used by QA and UI layers."""

    conversation_id: str
    summary: str
    key_points: list[str]
    risks: list[str]
    follow_ups: list[str]


class QualityScore(BaseModel):
    """Structured QA rubric returned by the quality scoring agent."""

    conversation_id: str
    professionalism: int = Field(..., ge=1, le=5)
    empathy: int = Field(..., ge=1, le=5)
    resolution: int = Field(..., ge=1, le=5)
    compliance: int = Field(..., ge=1, le=5)
    overall: int = Field(..., ge=1, le=5)
    summary_feedback: str
    risks: list[str]

    @model_validator(mode="after")
    def compute_overall(self) -> "QualityScore":
        if not self.overall:
            total = self.professionalism + self.empathy + self.resolution + self.compliance
            self.overall = round(total / 4)
        return self


def normalize_transcript_text(text: str) -> str:
    """Normalize whitespace and basic formatting for downstream tasks."""

    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def ensure_file(path: Path) -> Path:
    """Ensure that the provided path exists on disk."""

    if not path.exists():
        msg = f"File not found at {path}"
        raise FileNotFoundError(msg)
    return path
