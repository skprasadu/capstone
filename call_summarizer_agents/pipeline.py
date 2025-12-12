"""High-level pipeline for converting calls to insights."""
from __future__ import annotations

from typing import Any, Dict

from call_summarizer_agents.agents.routing_agent import RoutingAgent
from call_summarizer_agents.config.settings import AppSettings, load_settings


class CallSummarizationPipeline:
    """Bundle the routing agent with a friendly API surface."""

    def __init__(
        self,
        routing_agent: RoutingAgent | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.routing_agent = routing_agent or RoutingAgent(settings=self.settings)

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run the end-to-end pipeline."""

        return self.routing_agent.run(payload)
