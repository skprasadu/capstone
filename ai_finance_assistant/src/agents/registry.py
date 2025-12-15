from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AgentProfile:
    """Metadata describing a specialized finance agent."""

    name: str
    description: str
    responsibilities: List[str]
    routing_keywords: List[str]
    output_format: str
    safety_notes: List[str] = field(default_factory=list)

    def matches(self, topic: str) -> bool:
        topic_lower = topic.lower()
        return any(keyword in topic_lower for keyword in self.routing_keywords)


def build_registry() -> Dict[str, AgentProfile]:
    """Return the configured agents keyed by an identifier."""

    return {
        "finance_qa": AgentProfile(
            name="Finance Q&A Agent",
            description="Provides general-purpose financial education and definitions.",
            responsibilities=[
                "Answer basic financial literacy questions",
                "Explain terminology in approachable language",
                "Offer links to relevant educational resources",
            ],
            routing_keywords=["what is", "define", "explain", "how do", "difference"],
            output_format="Bulleted explanations with short examples and citations",
            safety_notes=["Avoid prescriptive investment recommendations."],
        ),
        "portfolio": AgentProfile(
            name="Portfolio Analysis Agent",
            description="Reviews user-provided holdings and risk preferences.",
            responsibilities=[
                "Summarize diversification and asset allocation",
                "Highlight concentration risk",
                "Map holdings to typical risk tolerance bands",
            ],
            routing_keywords=["portfolio", "allocation", "diversification", "holdings", "rebalance"],
            output_format="Table of metrics plus a concise narrative summary",
            safety_notes=["Explicitly state results are educational, not individualized advice."],
        ),
        "market": AgentProfile(
            name="Market Analysis Agent",
            description="Shares market context and trend highlights using cached data feeds.",
            responsibilities=[
                "Summarize index and sector movements",
                "Call out unusual volatility or volume",
                "Provide risk-aware takeaways for beginners",
            ],
            routing_keywords=["market", "index", "sector", "volatility", "trend", "macro"],
            output_format="Headline-style notes with percentage changes and source citations",
            safety_notes=["Avoid forward-looking predictions; focus on observed data."],
        ),
        "goals": AgentProfile(
            name="Goal Planning Agent",
            description="Guides users through setting time-bound financial goals.",
            responsibilities=[
                "Clarify time horizon, budget, and risk appetite",
                "Map goals to savings or investment vehicles",
                "Provide next-step checklists",
            ],
            routing_keywords=["goal", "plan", "timeline", "budget", "retirement", "college"],
            output_format="Step-by-step plan with milestones and links to education",
            safety_notes=["Encourage users to consult professionals for personal plans."],
        ),
        "news": AgentProfile(
            name="News Synthesizer Agent",
            description="Condenses financial news with context relevant to beginners.",
            responsibilities=[
                "Summarize articles in plain language",
                "Explain why the story matters to long-term investors",
                "Provide neutral, citation-backed commentary",
            ],
            routing_keywords=["news", "headline", "article", "update", "report"],
            output_format="3-5 bullet digest with source links",
            safety_notes=["Avoid sensationalism and keep language measured."],
        ),
        "tax": AgentProfile(
            name="Tax Education Agent",
            description="Explains tax-advantaged accounts and filing basics (not tax advice).",
            responsibilities=[
                "Define key account types (401k, IRA, HSA)",
                "Clarify contribution limits and timelines",
                "Outline questions to ask a professional",
            ],
            routing_keywords=["tax", "ira", "401k", "hsa", "deduction", "withholding"],
            output_format="FAQ-style responses with references to official sources",
            safety_notes=[
                "Remind users to consult certified tax professionals",
                "Do not offer jurisdiction-specific filing advice",
            ],
        ),
    }


def select_agent(topic: str) -> AgentProfile:
    """Route to the most relevant agent based on keyword matching."""

    registry = build_registry()
    for agent in registry.values():
        if agent.matches(topic):
            return agent
    return registry["finance_qa"]
