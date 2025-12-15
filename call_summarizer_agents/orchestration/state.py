from __future__ import annotations

from operator import add
from typing import Any, Annotated
from typing_extensions import TypedDict


class RunRecord(TypedDict):
    at: str
    conversation_id: str
    agent_name: str | None
    customer_name: str | None
    channel: str
    summary: str
    overall: int | None

class CallState(TypedDict, total=False):
    # Input
    raw_payload: dict[str, Any]

    # Validated/normalized (from IntakeAgent)
    intake: dict[str, Any]
    metadata: dict[str, Any]

    # Agent outputs
    transcript: dict[str, Any]
    summary: dict[str, Any]
    quality: dict[str, Any]

    # Memory: append-only list across runs for the same thread_id
    # Reducer = list concat
    runs: Annotated[list[RunRecord], add]

    # Final pipeline output (what your UI expects)
    result: dict[str, Any]