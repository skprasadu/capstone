from dataclasses import dataclass
from typing import List

from ai_finance_assistant.src.core.disclaimers import attach_disclaimer


@dataclass
class RetrievedDocument:
    title: str
    url: str
    summary: str


def retrieve(query: str) -> List[RetrievedDocument]:
    """Placeholder retrieval that returns static documents."""

    return [
        RetrievedDocument(
            title="Investing 101",
            url="https://example.com/investing-101",
            summary="Foundational principles of long-term investing and diversification.",
        ),
        RetrievedDocument(
            title="Understanding Risk Tolerance",
            url="https://example.com/risk",
            summary="How time horizon and volatility preferences influence allocations.",
        ),
    ]


def generate_response(query: str) -> str:
    docs = retrieve(query)
    bullets = "\n".join(f"- {doc.title}: {doc.summary} ({doc.url})" for doc in docs)
    message = f"Here are learning resources related to your question:\n{bullets}"
    return attach_disclaimer(message)
