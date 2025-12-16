from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List
import os

import chromadb
from openai import OpenAI

from ai_finance_assistant.src.core.disclaimers import attach_disclaimer
from ai_finance_assistant.src.data.knowledge_base import seed_articles
from ai_finance_assistant.src.utils.config_loader import load_config


@dataclass
class RetrievedDocument:
    title: str
    url: str
    summary: str


def _fallback_docs() -> List[RetrievedDocument]:
    docs: list[RetrievedDocument] = []
    for a in seed_articles():
        snippet = a.content.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        docs.append(RetrievedDocument(title=a.title, url=a.url, summary=snippet))
    return docs


def _openai_client() -> OpenAI | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


def _embed(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


def _chroma_params(cfg: dict[str, Any]) -> tuple[str, int, str, int, str]:
    rag = cfg.get("rag", {}) or {}
    host = os.getenv("CHROMA_HOST") or rag.get("chroma_host") or "localhost"
    port = int(os.getenv("CHROMA_PORT") or rag.get("chroma_port") or 8000)
    collection = os.getenv("CHROMA_COLLECTION") or rag.get("collection") or "finance_kb"
    top_k = int(rag.get("top_k") or 5)
    embed_model = rag.get("embedder") or "text-embedding-3-small"
    return host, port, collection, top_k, embed_model


def retrieve(query: str) -> List[RetrievedDocument]:
    cfg = load_config()
    rag = cfg.get("rag", {}) or {}

    if (rag.get("vector_store") or "").lower() != "chroma":
        return _fallback_docs()

    oa = _openai_client()
    if oa is None:
        return _fallback_docs()

    host, port, collection_name, top_k, embed_model = _chroma_params(cfg)

    try:
        chroma = chromadb.HttpClient(host=host, port=port)
        collection = chroma.get_collection(collection_name)

        q_emb = _embed(oa, embed_model, [query])[0]
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        metadatas = (res.get("metadatas") or [[]])[0] or []
        documents = (res.get("documents") or [[]])[0] or []

        hits: list[RetrievedDocument] = []
        for md, doc in zip(metadatas, documents):
            md = md or {}
            text = (doc or "").strip().replace("\n", " ")
            if len(text) > 200:
                text = text[:200] + "..."
            hits.append(
                RetrievedDocument(
                    title=md.get("title") or "Untitled",
                    url=md.get("url") or "",
                    summary=text,
                )
            )

        return hits or _fallback_docs()

    except Exception:
        # Chroma down, collection missing, schema mismatch, etc.
        return _fallback_docs()


def generate_response(query: str) -> str:
    docs = retrieve(query)
    bullets = "\n".join(f"- {doc.title}: {doc.summary} ({doc.url})" for doc in docs)
    message = f"Here are learning resources related to your question:\n{bullets}"
    return attach_disclaimer(message)