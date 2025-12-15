"""Entry point for running the call summarization pipeline locally."""
from __future__ import annotations

from pathlib import Path

from call_summarizer_agents.pipeline import CallSummarizationPipeline


def main() -> None:
    transcript_path = Path("call_summarizer_agents/data/sample_transcripts/sample_call.txt")
    transcript = transcript_path.read_text(encoding="utf-8")

    payload = {
        "conversation_id": "sample-call-001",
        "agent_name": "Jamie Agent",
        "customer_name": "Alex Customer",
        "channel": "voice",
        "audio_path": transcript_path,
        "transcript": transcript,
    }

    pipeline = CallSummarizationPipeline()
    result = pipeline.run(payload)

    print("Metadata:\n", result["metadata"])
    print("\nSummary:\n", result["summary"]["summary"])
    print("\nQuality:\n", result["quality"])


if __name__ == "__main__":
    main()
