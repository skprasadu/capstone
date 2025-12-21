from __future__ import annotations

import os
import pytest

from capstone_common.testing.deepeval import (
    configure_deepeval_openai_env,
    deepeval_pytestmark,
)

pytestmark = deepeval_pytestmark()
configure_deepeval_openai_env()

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