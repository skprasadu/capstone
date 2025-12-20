from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---- Opt-in guard (prevents token spend on normal `pytest`) ----
DEEPEVAL_RUN = os.getenv("DEEPEVAL_RUN", "0").lower() in ("1", "true", "yes", "y")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

pytestmark = pytest.mark.skipif(
    not (DEEPEVAL_RUN and OPENAI_KEY),
    reason="DeepEval tests are opt-in. Set DEEPEVAL_RUN=1 and OPENAI_API_KEY.",
)

# Help DeepEval pick OpenAI as the judge model (safe defaults)
if OPENAI_KEY:
    os.environ.setdefault("USE_OPENAI_MODEL", "1")
    os.environ.setdefault("OPENAI_MODEL_NAME", os.getenv("DEEPEVAL_OPENAI_MODEL", "gpt-4o-mini"))

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from call_summarizer_agents.agents.summarization_agent import SummarizationAgent


def test_call_summary_geval_correctness():
    transcript_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "sample_transcripts"
        / "sample_call.txt"
    )
    transcript = transcript_path.read_text(encoding="utf-8").strip()

    summarizer = SummarizationAgent(
        openai_api_key=OPENAI_KEY,
        openai_model=os.getenv("DEEPEVAL_OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
    )

    summary_payload = summarizer({"conversation_id": "deepeval-call-sample", "transcript": transcript})
    actual_summary = summary_payload.summary or ""
    assert actual_summary.strip(), "Summarizer returned an empty summary"

    expected_summary = """- Customer missed a delivery and wants to reschedule.
- Agent asks for and receives a tracking number.
- Agent schedules a redelivery for tomorrow (1 PM–5 PM) and customer agrees.
- Call ends politely with no further issues."""

    metric = GEval(
        name="Call summary correctness",
        criteria=(
            "The summary must capture the key facts:\n"
            "- missed delivery + reschedule request\n"
            "- agent requests tracking number\n"
            "- redelivery scheduled for tomorrow between 1 PM and 5 PM, customer agrees\n"
            "- polite closing / no additional issues\n"
            "Do NOT hallucinate refunds, cancellations, or policy violations."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=os.getenv("DEEPEVAL_OPENAI_MODEL", "gpt-4o-mini"),
    )

    test_case = LLMTestCase(
        input=transcript,
        actual_output=actual_summary,
        expected_output=expected_summary,
    )

    assert_test(test_case, [metric])