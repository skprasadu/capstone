# AI Finance Assistant

Prototype multi-agent finance education assistant with:
- keyword-based agent routing (6 agents)
- a simple RAG pipeline (Chroma-backed when enabled)
- Streamlit UI

> Educational only. Not financial/tax/investment advice.

---

## TL;DR Quick Test

1) Create `.env` at repo root:
```bash
OPENAI_API_KEY=YOUR_OPENAI_KEY
ALPHA_VANTAGE_API_KEY=YOUR_ALPHA_VANTAGE_KEY   # optional unless testing market API
```

2) Install deps (local dev):
```bash
pip install -e .
pip install chromadb pyyaml
```

3) Ingest the seed knowledge base into Chroma:
```bash
python -m ai_finance_assistant.src.rag.ingest --reset
```

4) Run Streamlit:
```bash
streamlit run ai_finance_assistant/src/web_app/streamlit_app.py --server.port=8502
```

5) Try queries (examples at bottom of this README).

---

## Project Layout

```
ai_finance_assistant/
  config.yaml
  README.md
  requirements.txt
  src/
    agents/          # routing + agent profiles
    core/            # disclaimers + blueprint
    data/            # seed knowledge base
    rag/             # retrieval + ingestion
    utils/           # config loader
    web_app/         # streamlit UI
    workflow/        # router (select agent + call RAG)
  tests/
```

---

## Configuration

Config is in `ai_finance_assistant/config.yaml`.

### LLM config
```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.2
  max_tokens: 512
```

### RAG / Chroma config
Add these keys (recommended) so ingestion + retrieval know where to persist the DB:
```yaml
rag:
  vector_store: "chroma"
  persist_dir: "ai_finance_assistant/.chroma"
  collection: "finance_kb"
  embedder: "text-embedding-3-small"
  top_k: 5
```

### Market data API config (Alpha Vantage)
```yaml
market_data:
  provider: "alpha_vantage"
  api_key: "YOUR_ALPHA_VANTAGE_KEY"
  fallback_provider: "yfinance"
```

**Recommended:** do NOT commit API keys into YAML. Use `.env` and pass keys via env vars; see “Finance API setup” below.

---

## Finance API setup (Alpha Vantage)

### Option A (fastest): put the key directly in config.yaml
Edit:
```yaml
market_data:
  api_key: "YOUR_REAL_KEY_HERE"
```

### Option B (recommended): keep key in `.env`
In `.env`:
```bash
ALPHA_VANTAGE_API_KEY=YOUR_REAL_KEY
```

Then add a tiny override in code (recommended) so config can stay key-less.

**Add this to** `ai_finance_assistant/src/utils/config_loader.py` (after loading YAML):
```python
import os

def load_config(path: Path | None = None) -> Dict[str, Any]:
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open() as handle:
        config = yaml.safe_load(handle)

    # Optional env overrides (keeps secrets out of git)
    av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if av_key and config.get("market_data", {}).get("api_key") in (None, "", "YOUR_ALPHA_VANTAGE_KEY"):
        config["market_data"]["api_key"] = av_key

    return config
```

### Quick “does my key work?” test (no code)
```bash
python - << 'PY'
import os, json, urllib.parse, urllib.request

key = os.getenv("ALPHA_VANTAGE_API_KEY")
if not key:
    raise SystemExit("Set ALPHA_VANTAGE_API_KEY in .env first")

params = urllib.parse.urlencode({
    "function": "GLOBAL_QUOTE",
    "symbol": "IBM",
    "apikey": key,
})
url = "https://www.alphavantage.co/query?" + params
print("GET", url)

with urllib.request.urlopen(url) as r:
    data = json.loads(r.read().decode("utf-8"))
print(json.dumps(data, indent=2)[:1200])
PY
```

**Notes**
- Alpha Vantage free tier is rate-limited (you may see throttling if you spam requests).
- The assistant UI currently does not call Alpha Vantage yet; this is just to validate config.

---

## What data are we ingesting into Chroma?

For this prototype we ingest a **small, curated seed knowledge base** (no scraping).
The seed KB lives in:
- `ai_finance_assistant/src/data/knowledge_base.py`

Each “Article” becomes **one vector document** with:
- `title`
- `category`
- `url`
- `summary` (short text used for embeddings and retrieval)

Default seed topics:
1) Diversification basics
2) Market indices basics
3) Tax-advantaged accounts overview

**Why this is the right demo data**
- small and deterministic
- no external fetches required
- easy to extend by adding more `seed_articles()`

---

## Enable Chroma ingestion + retrieval

### 1) Add/update the seed KB schema (summary field)
Edit `ai_finance_assistant/src/data/knowledge_base.py` to include a `summary`.

Example:
```python
from dataclasses import dataclass
from typing import List

@dataclass
class Article:
    title: str
    category: str
    url: str
    summary: str

def seed_articles() -> List[Article]:
    return [
        Article(
            title="Why diversification matters",
            category="portfolio_basics",
            url="https://example.com/diversification",
            summary="Diversification reduces single-asset risk by spreading exposure across many holdings and asset classes.",
        ),
        Article(
            title="Market indices explained",
            category="market_fundamentals",
            url="https://example.com/indices",
            summary="Market indices (like S&P 500) track a basket of securities to represent a segment of the market.",
        ),
        Article(
            title="Tax-advantaged accounts overview",
            category="tax_education",
            url="https://example.com/tax-accounts",
            summary="Accounts like 401(k), IRA, and HSA provide tax benefits that can improve long-term saving outcomes.",
        ),
    ]
```

### 2) Add an ingestion module
Create: `ai_finance_assistant/src/rag/ingest.py`

```python
from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

import chromadb
from openai import OpenAI

from ai_finance_assistant.src.data.knowledge_base import seed_articles
from ai_finance_assistant.src.utils.config_loader import load_config


def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete the collection before ingesting")
    args = parser.parse_args()

    cfg = load_config(Path("ai_finance_assistant/config.yaml"))
    rag_cfg = cfg.get("rag", {})

    persist_dir = Path(rag_cfg.get("persist_dir", "ai_finance_assistant/.chroma")).resolve()
    collection_name = rag_cfg.get("collection", "finance_kb")
    embedder = rag_cfg.get("embedder", "text-embedding-3-small")

    persist_dir.mkdir(parents=True, exist_ok=True)

    chroma = chromadb.PersistentClient(path=str(persist_dir))
    if args.reset:
        try:
            chroma.delete_collection(collection_name)
        except Exception:
            pass

    col = chroma.get_or_create_collection(name=collection_name)

    articles = seed_articles()

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    for a in articles:
        ids.append(f"{a.category}-{uuid4().hex[:8]}")
        # document text = what we embed (keep it small + relevant for the demo)
        docs.append(f"{a.title}\n\n{a.summary}")
        metas.append(
            {
                "title": a.title,
                "category": a.category,
                "url": a.url,
                "summary": a.summary,
            }
        )

    embeddings = embed_texts(docs, embedder)

    # Prefer upsert to avoid duplicate-id pain when re-running
    if hasattr(col, "upsert"):
        col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    else:
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    print(f"[ok] Ingested {len(ids)} docs into '{collection_name}' at '{persist_dir}'")


if __name__ == "__main__":
    main()
```

Run it:
```bash
python -m ai_finance_assistant.src.rag.ingest --reset
```

### 3) Update retrieval to use Chroma (instead of the static stub)
Edit `ai_finance_assistant/src/rag/pipeline.py` to query Chroma when available:

```python
from dataclasses import dataclass
from typing import List
from pathlib import Path

from openai import OpenAI

from ai_finance_assistant.src.core.disclaimers import attach_disclaimer
from ai_finance_assistant.src.utils.config_loader import load_config


@dataclass
class RetrievedDocument:
    title: str
    url: str
    summary: str


def _fallback_retrieve(_: str) -> List[RetrievedDocument]:
    # Existing behavior (keeps the app working without Chroma)
    return [
        RetrievedDocument(
            title="Investing 101",
            url="https://example.com/investing-101",
            summary="Foundational principles of long-term investing and diversification.",
        ),
        RetrievedDocument(
            title="Understanding Risk Tolerance",
            url="https://example.com/risk",
            summary="How time horizon and volatility preferences influence allocations.",
        ),
    ]


def retrieve(query: str) -> List[RetrievedDocument]:
    cfg = load_config(Path("ai_finance_assistant/config.yaml"))
    rag_cfg = cfg.get("rag", {})

    persist_dir = rag_cfg.get("persist_dir", "ai_finance_assistant/.chroma")
    collection_name = rag_cfg.get("collection", "finance_kb")
    embedder = rag_cfg.get("embedder", "text-embedding-3-small")
    top_k = int(rag_cfg.get("top_k", 5))

    # Lazy import so the UI still runs without chromadb installed
    try:
        import chromadb
    except Exception:
        return _fallback_retrieve(query)

    if not query.strip():
        return []

    # Embed query with OpenAI embeddings (requires OPENAI_API_KEY)
    try:
        client = OpenAI()
        q_emb = client.embeddings.create(model=embedder, input=[query]).data[0].embedding
    except Exception:
        return _fallback_retrieve(query)

    try:
        chroma = chromadb.PersistentClient(path=str(persist_dir))
        col = chroma.get_or_create_collection(name=collection_name)
        res = col.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["metadatas"],
        )
        metas = (res.get("metadatas") or [[]])[0]

        out: List[RetrievedDocument] = []
        for m in metas:
            if not m:
                continue
            out.append(
                RetrievedDocument(
                    title=str(m.get("title") or ""),
                    url=str(m.get("url") or ""),
                    summary=str(m.get("summary") or ""),
                )
            )
        return out or _fallback_retrieve(query)
    except Exception:
        return _fallback_retrieve(query)


def generate_response(query: str) -> str:
    docs = retrieve(query)
    bullets = "\n".join(f"- {doc.title}: {doc.summary} ({doc.url})" for doc in docs)
    message = f"Here are learning resources related to your question:\n{bullets}"
    return attach_disclaimer(message)
```

---

## Sample Queries (for Streamlit)

These test **routing** (which agent is selected) + **retrieval** (Chroma results).

### Finance Q&A Agent (keywords like “what is”, “define”, “explain”)
- `What is diversification and why does it matter?`
- `Explain what an index fund is.`
- `Define risk tolerance in investing.`

### Portfolio Analysis Agent (keywords like “portfolio”, “allocation”, “holdings”)
- `Can you review my portfolio allocation? 70% VTI, 20% VXUS, 10% BND`
- `Is my portfolio too concentrated in tech?`

### Market Analysis Agent (keywords like “market”, “index”, “sector”, “volatility”)
- `What does market volatility mean?`
- `Explain what the S&P 500 represents.`

### Goal Planning Agent (keywords like “goal”, “plan”, “timeline”, “retirement”)
- `Help me plan a 5-year savings goal for a home down payment.`
- `How should I think about retirement planning timelines?`

### News Synthesizer Agent (keywords like “news”, “headline”, “report”)
- `Summarize the latest market news and explain why it matters.`
  (Note: this prototype does not fetch real news yet; it will still respond with educational docs.)

### Tax Education Agent (keywords like “tax”, “401k”, “ira”, “hsa”)
- `What’s the difference between a 401(k) and an IRA?`
- `Explain what an HSA is and how it works.`

---

## Notes / Current Prototype Limitations

- Market data provider config exists, but the Streamlit UI does not yet call Alpha Vantage.
- RAG defaults to a fallback stub unless Chroma + OpenAI embeddings are configured.
- The seed KB is intentionally small and deterministic; extend by adding more `seed_articles()`.