from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request


def global_quote(symbol: str) -> dict:
    key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not key:
        return {"error": "Missing ALPHA_VANTAGE_API_KEY"}

    params = urllib.parse.urlencode({"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": key})
    url = "https://www.alphavantage.co/query?" + params

    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode("utf-8"))