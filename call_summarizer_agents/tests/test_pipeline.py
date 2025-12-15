from pathlib import Path
from call_summarizer_agents.config.settings import AppSettings
from call_summarizer_agents.pipeline import CallSummarizationPipeline


SAMPLE_TRANSCRIPT = Path("../call_summarizer_agents/data/sample_transcripts/sample_call.txt")


def test_pipeline_runs_with_sample_transcript():
    transcript = SAMPLE_TRANSCRIPT.read_text(encoding="utf-8")
    payload = {
        "conversation_id": "sample-call-pytest",
        "agent_name": "Py Test",
        "customer_name": "QA Bot",
        "channel": "voice",
        "audio_path": SAMPLE_TRANSCRIPT,
        "transcript": transcript,
    }
    settings = AppSettings(openai_api_key=None, whisper_api_key=None)
    pipeline = CallSummarizationPipeline(settings=settings)
    result = pipeline.run(payload)

    assert set(result.keys()) == {"metadata", "transcript", "summary", "quality"}
    assert result["summary"]["conversation_id"] == "sample-call-pytest"
    assert result["quality"]["overall"] >= 1
    assert "summary" in result["summary"]
