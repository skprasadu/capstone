from __future__ import annotations

import os
from hashlib import sha1
from typing import Any

import chromadb
from openai import OpenAI

from ai_finance_assistant.src.data.knowledge_base import seed_articles
from ai_finance_assistant.src.utils.config_loader import load_config
from capstone_common.llm.openai_client import require_openai_client_from_env

def _get_cfg() -> dict[str, Any]:
    return load_config()


def _chroma_params(cfg: dict[str, Any]) -> tuple[str, int, str]:
    rag = cfg.get("rag", {}) or {}
    host = os.getenv("CHROMA_HOST") or rag.get("chroma_host") or "localhost"
    port = int(os.getenv("CHROMA_PORT") or rag.get("chroma_port") or 8000)
    collection = os.getenv("CHROMA_COLLECTION") or rag.get("collection") or "finance_kb"
    return host, port, collection


def _openai_client() -> OpenAI:
    # Keep behavior: hard fail if missing
    client = require_openai_client_from_env("OPENAI_API_KEY", wrap_langsmith=False)
    return client


def _embed(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


def main() -> None:
    cfg = _get_cfg()
    rag = cfg.get("rag", {}) or {}

    host, port, collection_name = _chroma_params(cfg)
    embed_model = rag.get("embedder") or "text-embedding-3-small"

    chroma = chromadb.HttpClient(host=host, port=port)
    # cosine is usually the right space for embedding similarity
    collection = chroma.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    articles = seed_articles()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for a in articles:
        doc_id = "seed-" + sha1(f"{a.title}|{a.category}".encode("utf-8")).hexdigest()[:12]
        ids.append(doc_id)
        documents.append(a.content)
        metadatas.append({"title": a.title, "category": a.category, "url": a.url})

    oa = _openai_client()
    embeddings = _embed(oa, embed_model, documents)

    # Prefer upsert (idempotent). Fallback to delete+add if needed.
    try:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    except AttributeError:
        try:
            collection.delete(ids=ids)
        except Exception:
            pass
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    print(
        f"[ok] Ingested/updated {len(ids)} docs into collection '{collection_name}' "
        f"on {host}:{port} (embedder={embed_model})"
    )


if __name__ == "__main__":
    main()