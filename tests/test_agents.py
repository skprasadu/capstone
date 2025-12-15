import pytest

from call_summarizer_agents.agents.transcription_agent import TranscriptionAgent
from call_summarizer_agents.agents.summarization_agent import SummarizationAgent
from call_summarizer_agents.agents.quality_score_agent import QualityScoreAgent


def test_transcription_agent_prefers_provided_transcript():
    agent = TranscriptionAgent()
    payload = {
        "conversation_id": "conv-1",
        "transcript": " Hello world\nThis is a test ",
    }

    result = agent(payload)

    assert result.transcript == "Hello world\nThis is a test"
    assert result.audio_path is None


def test_transcription_agent_requires_input():
    agent = TranscriptionAgent()
    with pytest.raises(ValueError):
        agent({"conversation_id": "conv-2"})


def test_summarization_agent_fallback_and_key_extraction():
    transcript = "Customer requested a refund. Agent will email tomorrow."
    agent = SummarizationAgent()

    result = agent({"transcript": transcript})

    assert "Auto-generated summary" in result.summary
    assert result.key_points == ["Customer requested a refund", "Agent will email tomorrow"]
    assert "refund" in " ".join(result.risks)
    assert any("email" in followup for followup in result.follow_ups)


def test_quality_score_agent_scores_within_bounds():
    transcript = "Thank you for calling. We will verify your account and send a solution."
    agent = QualityScoreAgent()

    result = agent({"transcript": transcript, "conversation_id": "conv-3"})

    for field in ("professionalism", "empathy", "resolution", "compliance", "overall"):
        value = getattr(result, field)
        assert 1 <= value <= 5
