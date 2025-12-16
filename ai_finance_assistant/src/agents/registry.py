from dataclasses import dataclass, field
from typing import Dict, List, Tuple


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
        topic_lower = (topic or "").lower()
        return any(keyword in topic_lower for keyword in self.routing_keywords)


def build_registry() -> Dict[str, AgentProfile]:
    """Return the configured agents keyed by an identifier."""

    return {
        # IMPORTANT: finance_qa is the fallback; do NOT let it steal tax/portfolio/etc.
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
                "Explain volatility and macro terms",
                "Summarize broad market context",
                "Provide risk-aware takeaways for beginners",
            ],
            # NOTE: removed generic "index" so "What is an index fund?" goes to finance_qa
            routing_keywords=["market", "sector", "volatility", "trend", "macro", "s&p", "nasdaq", "dow"],
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
            routing_keywords=["goal", "plan", "timeline", "budget", "retirement", "college", "down payment"],
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
        "stock_quote": AgentProfile(
            name="Stock Quote Agent",
            description="Fetches the latest stock quote using Alpha Vantage.",
            responsibilities=[
                "Extract ticker symbol from the query",
                "Fetch quote via Alpha Vantage (GLOBAL_QUOTE)",
                "Return a short, factual snapshot",
            ],
            routing_keywords=["stock price", "quote", "current price", "price of"],
            output_format="Short quote summary + raw JSON",
            safety_notes=["Quotes may be delayed and rate-limited. Not investment advice."],
        ),
    }


def select_agent_with_id(topic: str) -> Tuple[str, AgentProfile]:
    """
    Deterministic routing (simple + predictable):
    - Prefer specialized agents first
    - finance_qa is the fallback
    """
    registry = build_registry()

    # priority order matters
    ordered = ["tax", "portfolio", "news", "goals", "market", "finance_qa"]
    for agent_id in ordered:
        agent = registry[agent_id]
        if agent.matches(topic):
            return agent_id, agent

    return "finance_qa", registry["finance_qa"]


def select_agent(topic: str) -> AgentProfile:
    """Backwards-compatible helper returning only the profile."""
    _, agent = select_agent_with_id(topic)
    return agent