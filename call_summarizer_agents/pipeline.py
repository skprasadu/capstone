"""High-level pipeline for converting calls to insights."""
from __future__ import annotations

from typing import Any, Dict

from call_summarizer_agents.agents.routing_agent import RoutingAgent


class CallSummarizationPipeline:
    """Bundle the routing agent with a friendly API surface."""

    def __init__(self, routing_agent: RoutingAgent | None = None) -> None:
        self.routing_agent = routing_agent or RoutingAgent()

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run the end-to-end pipeline."""

        return self.routing_agent.run(payload)
