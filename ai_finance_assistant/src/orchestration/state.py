from __future__ import annotations

from operator import add
from typing import Any, Annotated
from typing_extensions import TypedDict


class RunRecord(TypedDict):
    at: str
    conversation_id: str
    agent_id: str
    agent_name: str
    query: str
    symbol: str | None


class FinanceState(TypedDict, total=False):
    raw_payload: dict[str, Any]

    route: dict[str, Any]
    retrieval: dict[str, Any]
    market: dict[str, Any]

    result: dict[str, Any]

    # memory (append-only)
    runs: Annotated[list[RunRecord], add]