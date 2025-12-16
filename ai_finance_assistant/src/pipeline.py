from __future__ import annotations

from typing import Any, Dict

from ai_finance_assistant.src.orchestration.graph import FinanceAssistantGraph


class FinanceAssistantPipeline:
    """Friendly API around the LangGraph workflow (keeps UI dumb)."""

    _graph_runner: FinanceAssistantGraph | None = None

    def __init__(self) -> None:
        if FinanceAssistantPipeline._graph_runner is None:
            FinanceAssistantPipeline._graph_runner = FinanceAssistantGraph()
        self.graph_runner = FinanceAssistantPipeline._graph_runner

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.graph_runner.run(payload)

    def list_conversations(self):
        return self.graph_runner.list_conversations()

    def get_runs(self, conversation_id: str):
        return self.graph_runner.get_runs(conversation_id)

    def get_latest_result(self, conversation_id: str):
        return self.graph_runner.get_latest_result(conversation_id)