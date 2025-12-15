from __future__ import annotations

import json
import os
from typing import Any

DEBUG = os.getenv("CALL_SUMMARIZER_DEBUG", "0").lower() in ("1", "true", "yes", "y")


def _mask(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    if len(s) <= 12:
        return "***"
    return f"{s[:6]}...{s[-4:]}"


def dlog(event: str, **fields: Any) -> None:
    if not DEBUG:
        return

    safe: dict[str, Any] = {}
    for k, v in fields.items():
        if "key" in k.lower() or "token" in k.lower():
            safe[k] = _mask(v)
        else:
            safe[k] = v

    print(f"[CALL_SUMMARIZER_DEBUG] {event} {json.dumps(safe, default=str)}", flush=True)