"""High-level pipeline for converting calls to insights."""
from __future__ import annotations

from typing import Any, Dict

from call_summarizer_agents.config.settings import AppSettings, load_settings
from call_summarizer_agents.orchestration.graph import CallSummarizerGraph
from call_summarizer_agents.orchestration.state import RunRecord

class CallSummarizationPipeline:
    """Friendly API surface around the LangGraph workflow."""

    _graph_runner: CallSummarizerGraph | None = None

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or load_settings()

        # Keep one graph+checkpointer per process so "memory" works across runs.
        if CallSummarizationPipeline._graph_runner is None:
            CallSummarizationPipeline._graph_runner = CallSummarizerGraph(settings=self.settings)

        self.graph_runner = CallSummarizationPipeline._graph_runner

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.graph_runner.run(payload)

    # Optional helper methods (no UI logic required)
    def get_runs(self, conversation_id: str):
        return self.graph_runner.get_runs(conversation_id)

    def get_state_history(self, conversation_id: str):
        return self.graph_runner.get_state_history(conversation_id)
    
    def list_conversations(self) -> list[RunRecord]:
        return self.graph_runner.list_conversations()
    
    def get_latest_result(self, conversation_id: str):
        return self.graph_runner.get_latest_result(conversation_id)