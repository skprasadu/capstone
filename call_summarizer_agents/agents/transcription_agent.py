"""Transcription agent for converting audio into structured text."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from call_summarizer_agents.utils.validation import TranscriptPayload, ensure_file, normalize_transcript_text


class TranscriptionAgent:
    """Convert audio to text with fallback to existing transcripts."""

    def __init__(self, engine: str = "whisper") -> None:
        self.name = "TranscriptionAgent"
        self.engine = engine

    def __call__(self, payload: Dict[str, Any]) -> TranscriptPayload:
        transcript = payload.get("transcript")
        audio_path: Optional[str | Path] = payload.get("audio_path")
        conversation_id: str = payload.get("conversation_id", "unknown")

        if transcript:
            normalized = normalize_transcript_text(transcript)
            return TranscriptPayload(
                conversation_id=conversation_id,
                transcript=normalized,
                audio_path=Path(audio_path) if audio_path else None,
                duration_seconds=None,
            )

        if audio_path:
            audio_path = ensure_file(Path(audio_path))
            pseudo_transcript = self._pseudo_transcribe(audio_path)
            return TranscriptPayload(
                conversation_id=conversation_id,
                transcript=normalize_transcript_text(pseudo_transcript),
                audio_path=audio_path,
                duration_seconds=None,
            )

        msg = "No transcript or audio payload available for transcription."
        raise ValueError(msg)

    def _pseudo_transcribe(self, audio_path: Path) -> str:
        """Lightweight placeholder for Whisper/Deepgram integrations."""

        if audio_path.suffix.lower() == ".txt":
            return audio_path.read_text(encoding="utf-8")

        return (
            f"Transcription placeholder for {audio_path.name}. "
            "Integrate Whisper or Deepgram SDKs here."
        )
