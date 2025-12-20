from __future__ import annotations

import os
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

from ai_finance_assistant.src.core.disclaimers import FINANCE_DISCLAIMER
from ai_finance_assistant.src.pipeline import FinanceAssistantPipeline


def test_finance_answer_relevance_geval():
    pipeline = FinanceAssistantPipeline()

    query = "What is an index fund?"
    result = pipeline.run({"query": query})
    answer = result.get("answer") or ""

    # Cheap deterministic guardrail (then DeepEval for semantic quality)
    assert FINANCE_DISCLAIMER in answer, "Finance disclaimer missing from answer"

    metric = GEval(
        name="Finance answer relevance",
        criteria=(
            "Response should be educational and relevant to the user's question about index funds.\n"
            "It should suggest at least one learning resource relevant to index funds / market indices.\n"
            "It must not include prescriptive investment advice (no buy/sell/hold recommendations)."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
        model=os.getenv("DEEPEVAL_OPENAI_MODEL", "gpt-4o-mini"),
    )

    test_case = LLMTestCase(input=query, actual_output=answer)
    assert_test(test_case, [metric])