from dataclasses import dataclass
from typing import Dict, List

from ai_finance_assistant.src.agents.registry import AgentProfile, build_registry


@dataclass
class AssistantCapability:
    name: str
    description: str
    endpoints: List[str]


@dataclass
class AssistantBlueprint:
    name: str
    goals: List[str]
    capabilities: List[AssistantCapability]
    agents: Dict[str, AgentProfile]


def bootstrap_blueprint() -> AssistantBlueprint:
    capabilities = [
        AssistantCapability(
            name="Education-first conversations",
            description=(
                "Context-aware answers that keep disclaimers visible and link back to the knowledge base."
            ),
            endpoints=["/chat", "/knowledge"],
        ),
        AssistantCapability(
            name="Market-aware explanations",
            description=(
                "Lightweight market snapshots that reference cached data and avoid forward-looking statements."
            ),
            endpoints=["/market/snapshot"],
        ),
        AssistantCapability(
            name="Goal and portfolio walkthroughs",
            description=(
                "Interactive checklists that pair user goals with risk-aware educational steps."
            ),
            endpoints=["/goals", "/portfolio/analyze"],
        ),
    ]

    return AssistantBlueprint(
        name="AI Finance Assistant",
        goals=[
            "Democratize access to financial education",
            "Preserve context across multi-turn sessions",
            "Ground responses in vetted, cite-able content",
        ],
        capabilities=capabilities,
        agents=build_registry(),
    )
