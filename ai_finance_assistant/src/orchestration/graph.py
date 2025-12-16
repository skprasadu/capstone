from __future__ import annotations

import re
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from ai_finance_assistant.src.agents.registry import build_registry, select_agent_with_id
from ai_finance_assistant.src.core.disclaimers import attach_disclaimer
from ai_finance_assistant.src.market.alpha_vantage import global_quote
from ai_finance_assistant.src.orchestration.state import FinanceState, RunRecord
from ai_finance_assistant.src.rag.pipeline import retrieve as rag_retrieve


_TICKER_FROM_DOLLAR = re.compile(r"\$([A-Za-z]{1,6})\b")
_TICKER_FROM_PRICE_PHRASE = re.compile(
    r"\b(?:stock\s+price|price|quote)\s+(?:of\s+)?([A-Za-z]{1,6})\b", re.IGNORECASE
)
_TICKER_FROM_SUFFIX = re.compile(r"\b([A-Za-z]{1,6})\s+(?:stock\s+price|stock|quote)\b", re.IGNORECASE)


def _extract_ticker(query: str) -> str | None:
    q = (query or "").strip()
    if not q:
        return None

    m = _TICKER_FROM_DOLLAR.search(q)
    if m:
        return m.group(1).upper()

    m = _TICKER_FROM_PRICE_PHRASE.search(q)
    if m:
        return m.group(1).upper()

    m = _TICKER_FROM_SUFFIX.search(q)
    if m:
        return m.group(1).upper()

    return None


def _format_quote(raw: dict[str, Any], symbol: str) -> str:
    if not isinstance(raw, dict):
        return attach_disclaimer(f"Alpha Vantage returned an unexpected response for {symbol}.")

    if raw.get("error"):
        return attach_disclaimer(f"Stock quote failed for **{symbol}**: {raw['error']}")

    if "Note" in raw:
        return attach_disclaimer(f"Alpha Vantage rate limit: {raw.get('Note')}")

    if "Error Message" in raw:
        return attach_disclaimer(f"Alpha Vantage error for **{symbol}**: {raw.get('Error Message')}")

    quote = raw.get("Global Quote") or {}
    if not isinstance(quote, dict) or not quote:
        return attach_disclaimer(f"No quote data available for **{symbol}**.")

    price = quote.get("05. price")
    day = quote.get("07. latest trading day")
    change = quote.get("09. change")
    change_pct = quote.get("10. change percent")
    prev_close = quote.get("08. previous close")
    volume = quote.get("06. volume")

    lines = [
        f"**{symbol}** (Alpha Vantage GLOBAL_QUOTE)",
        f"- Price: `{price}`",
        f"- Change: `{change}` ({change_pct})",
        f"- Latest trading day: `{day}`",
        f"- Previous close: `{prev_close}`",
        f"- Volume: `{volume}`",
    ]
    return attach_disclaimer("\n".join(lines))


class FinanceAssistantGraph:
    """
    LangGraph orchestrator:
    - One query in, route + execute + persist run record
    - InMemorySaver keeps per-thread state during the process lifetime
    """

    def __init__(self, checkpointer: Any | None = None) -> None:
        self.checkpointer = checkpointer or InMemorySaver()
        self._conversation_index: dict[str, RunRecord] = {}

        workflow = self._build()
        self.graph = workflow.compile(checkpointer=self.checkpointer)

    def _build(self) -> StateGraph:
        g = StateGraph(FinanceState)

        g.add_node("intake", self._node_intake)
        g.add_node("route", self._node_route)
        g.add_node("execute", self._node_execute)
        g.add_node("finalize", self._node_finalize)

        g.add_edge(START, "intake")
        g.add_edge("intake", "route")
        g.add_edge("route", "execute")
        g.add_edge("execute", "finalize")
        g.add_edge("finalize", END)

        return g

    # -------------------------
    # Nodes
    # -------------------------

    def _node_intake(self, state: FinanceState) -> dict[str, Any]:
        raw = state.get("raw_payload") or {}

        conversation_id = str(raw.get("conversation_id") or "").strip()
        if not conversation_id:
            conversation_id = f"conv-{uuid4().hex[:8]}"

        query = str(raw.get("query") or "").strip()

        asked_at = datetime.now(timezone.utc).isoformat()

        return {
            "raw_payload": {"conversation_id": conversation_id, "query": query},
            "route": {},
            "retrieval": {},
            "market": {},
            "result": {},
            "metadata": {"conversation_id": conversation_id, "asked_at": asked_at},
        }

    def _node_route(self, state: FinanceState) -> dict[str, Any]:
        raw = state.get("raw_payload") or {}
        query = str(raw.get("query") or "").strip()

        registry = build_registry()

        # 1) Stock quote detection (requires ticker)
        symbol = _extract_ticker(query)
        if symbol:
            agent = registry.get("stock_quote")
            return {
                "route": {
                    "agent_id": "stock_quote",
                    "agent_name": agent.name if agent else "Stock Quote Agent",
                    "reason": f"Detected a stock quote request for symbol '{symbol}'.",
                    "symbol": symbol,
                }
            }

        # 2) Keyword routing for the rest
        agent_id, agent = select_agent_with_id(query)
        return {
            "route": {
                "agent_id": agent_id,
                "agent_name": agent.name,
                "reason": "Routed by keyword match (specialized agents first; finance_qa fallback).",
                "symbol": None,
            }
        }

    def _node_execute(self, state: FinanceState) -> dict[str, Any]:
        raw = state.get("raw_payload") or {}
        query = str(raw.get("query") or "").strip()

        route = state.get("route") or {}
        agent_id = route.get("agent_id") or "finance_qa"

        # Alpha Vantage path
        if agent_id == "stock_quote":
            symbol = route.get("symbol") or ""
            raw_quote = global_quote(symbol)
            answer = _format_quote(raw_quote, symbol)
            return {
                "market": {"provider": "alpha_vantage", "symbol": symbol, "raw": raw_quote},
                "retrieval": {"docs": []},
                "answer": answer,
            }

        # RAG path (Chroma-backed, with fallback inside rag.pipeline)
        docs = rag_retrieve(query)
        bullets = "\n".join(f"- {d.title}: {d.summary} ({d.url})" for d in docs)
        answer = attach_disclaimer(f"Here are learning resources related to your question:\n{bullets}")

        return {
            "retrieval": {"docs": [asdict(d) for d in docs]},
            "market": {},
            "answer": answer,
        }

    def _node_finalize(self, state: FinanceState) -> dict[str, Any]:
        raw = state.get("raw_payload") or {}
        route = state.get("route") or {}
        meta = state.get("metadata") or {}

        conversation_id = meta.get("conversation_id") or raw.get("conversation_id") or "unknown"
        asked_at = meta.get("asked_at") or datetime.now(timezone.utc).isoformat()

        result = {
            "metadata": {
                "conversation_id": conversation_id,
                "asked_at": asked_at,
            },
            "request": {
                "query": raw.get("query") or "",
            },
            "route": route,
            "retrieval": state.get("retrieval") or {},
            "market": state.get("market") or {},
            "answer": state.get("answer") or "",
        }

        run_record: RunRecord = {
            "at": asked_at,
            "conversation_id": conversation_id,
            "agent_id": str(route.get("agent_id") or "finance_qa"),
            "agent_name": str(route.get("agent_name") or "Finance Q&A Agent"),
            "query": str(raw.get("query") or "")[:160],
            "symbol": route.get("symbol"),
        }

        return {"result": result, "runs": [run_record]}

    # -------------------------
    # Public API
    # -------------------------

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        conversation_id = str(payload.get("conversation_id") or "").strip()
        if not conversation_id:
            conversation_id = f"conv-{uuid4().hex[:8]}"
        payload = {**payload, "conversation_id": conversation_id}

        config = {
            "configurable": {"thread_id": conversation_id},
            "tags": ["ai-finance-assistant"],
            "metadata": {"conversation_id": conversation_id},
            "run_name": "ai-finance-assistant-graph",
        }

        final_state: FinanceState = self.graph.invoke({"raw_payload": payload}, config)

        runs = list(final_state.get("runs") or [])
        if runs:
            self._conversation_index[conversation_id] = runs[-1]

        return final_state["result"]

    def list_conversations(self) -> list[RunRecord]:
        return sorted(self._conversation_index.values(), key=lambda r: r.get("at", ""), reverse=True)

    def get_runs(self, conversation_id: str) -> list[RunRecord]:
        config = {"configurable": {"thread_id": conversation_id}}
        snapshot = self.graph.get_state(config)
        return list(snapshot.values.get("runs") or [])

    def get_latest_result(self, conversation_id: str) -> dict[str, Any] | None:
        config = {"configurable": {"thread_id": conversation_id}}
        snapshot = self.graph.get_state(config)
        return snapshot.values.get("result") or None