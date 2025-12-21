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
    # Input
    raw_payload: dict[str, Any]

    # Added: these keys are used by nodes but were missing from the schema
    metadata: dict[str, Any]
    answer: str

    # Routing + outputs
    route: dict[str, Any]
    retrieval: dict[str, Any]
    market: dict[str, Any]

    # Final pipeline output
    result: dict[str, Any]

    # Memory (append-only)
    runs: Annotated[list[RunRecord], add]