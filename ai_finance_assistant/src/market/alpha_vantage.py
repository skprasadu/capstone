from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict


def global_quote(symbol: str) -> Dict[str, Any]:
    """
    Alpha Vantage GLOBAL_QUOTE.
    Reads API key from env: ALPHA_VANTAGE_API_KEY
    """
    key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not key:
        return {"error": "Missing ALPHA_VANTAGE_API_KEY"}

    symbol = (symbol or "").strip().upper()
    if not symbol:
        return {"error": "Missing symbol"}

    params = urllib.parse.urlencode({"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": key})
    url = "https://www.alphavantage.co/query?" + params

    try:
        with urllib.request.urlopen(url) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"error": f"Alpha Vantage request failed: {e!r}"}