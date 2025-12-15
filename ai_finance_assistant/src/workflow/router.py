from dataclasses import dataclass

from ai_finance_assistant.src.agents.registry import AgentProfile, select_agent
from ai_finance_assistant.src.rag.pipeline import generate_response


@dataclass
class RoutedResponse:
    agent: AgentProfile
    content: str
    reasoning: str


def route_query(user_query: str) -> RoutedResponse:
    agent = select_agent(user_query)
    content = generate_response(user_query)
    reasoning = (
        "Selected agent based on keyword match; RAG stub returns curated learning resources."
    )
    return RoutedResponse(agent=agent, content=content, reasoning=reasoning)
