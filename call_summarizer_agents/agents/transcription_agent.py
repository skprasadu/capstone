"""Transcription agent for converting audio into structured text."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

from call_summarizer_agents.utils.validation import TranscriptPayload, ensure_file, normalize_transcript_text
from call_summarizer_agents.utils.debug import dlog

class TranscriptionAgent:
    """Convert audio to text with fallback to existing transcripts."""

    def __init__(
        self,
        engine: str = "whisper",
        whisper_api_key: str | None = None,
        whisper_model: str = "whisper-1",
    ) -> None:
        self.name = "TranscriptionAgent"
        self.engine = engine
        self.whisper_model = whisper_model
        self._openai_client: Optional[OpenAI] = (
            OpenAI(api_key=whisper_api_key) if whisper_api_key else None
        )

    def __call__(self, payload: Dict[str, Any]) -> TranscriptPayload:
        transcript = payload.get("transcript")
        audio_path: Optional[str | Path] = payload.get("audio_path")
        conversation_id: str = payload.get("conversation_id", "unknown")
        dlog(
            "transcription.call",
            conversation_id=conversation_id,
            transcript_present=bool(transcript),
            audio_path=str(audio_path) if audio_path else None,
            openai_client_present=bool(self._openai_client),
            whisper_model=self.whisper_model,
        )

        if transcript:
            dlog("transcription.branch", branch="provided_transcript", transcript_len=len(transcript))

            normalized = normalize_transcript_text(transcript)
            return TranscriptPayload(
                conversation_id=conversation_id,
                transcript=normalized,
                audio_path=Path(audio_path) if audio_path else None,
                duration_seconds=None,
            )

        if audio_path:
            audio_path = ensure_file(Path(audio_path))
            dlog(
                "transcription.audio_file",
                path=str(audio_path),
                size_bytes=audio_path.stat().st_size,
                suffix=audio_path.suffix.lower(),
            )
            transcript_text = self._transcribe_audio(audio_path)
            return TranscriptPayload(
                conversation_id=conversation_id,
                transcript=normalize_transcript_text(transcript_text),
                audio_path=audio_path,
                duration_seconds=None,
            )

        msg = "No transcript or audio payload available for transcription."
        raise ValueError(msg)

    def _transcribe_audio(self, audio_path: Path) -> str:
        if audio_path.suffix.lower() == ".txt":
            dlog("transcription.txt_shortcut", path=str(audio_path))
            return audio_path.read_text(encoding="utf-8")

        if not self._openai_client:
            dlog("transcription.pseudo", reason="no_openai_client")
            return self._pseudo_transcribe(audio_path)

        dlog("transcription.whisper.start", path=str(audio_path), model=self.whisper_model)

        try:
            with audio_path.open("rb") as audio_file:
                response = self._openai_client.audio.transcriptions.create(
                    model=self.whisper_model,
                    file=audio_file,
                )
            text = str(getattr(response, "text", "") or "")
            dlog("transcription.whisper.ok", text_len=len(text))
            return text or self._pseudo_transcribe(audio_path)

        except Exception as e:
            dlog("transcription.whisper.error", error=repr(e))
            return self._pseudo_transcribe(audio_path)

    def _pseudo_transcribe(self, audio_path: Path) -> str:
        """Lightweight placeholder for Whisper/Deepgram integrations."""

        return (
            f"Transcription placeholder for {audio_path.name}. "
            "Integrate Whisper or Deepgram SDKs here."
        )
