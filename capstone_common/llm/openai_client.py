from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

try:
    from langsmith.wrappers import wrap_openai as _wrap_openai
except Exception:
    _wrap_openai = None


def get_openai_client(
    api_key: str | None = None,
    *,
    client: Any | None = None,
    wrap_langsmith: bool = True,
) -> Any | None:
    """
    Return an OpenAI client, optionally wrapped for LangSmith tracing.

    - If `client` is provided, it will be returned (optionally wrapped).
    - If `client` is None, a new OpenAI client is created if `api_key` is provided.
    - Wrapping is best-effort and will never raise.
    """
    if client is None:
        if not api_key:
            return None
        client = OpenAI(api_key=api_key)

    if wrap_langsmith and _wrap_openai:
        try:
            client = _wrap_openai(client)
        except Exception:
            pass

    return client


def get_openai_client_from_env(
    env_var: str = "OPENAI_API_KEY",
    *,
    wrap_langsmith: bool = True,
) -> Any | None:
    return get_openai_client(os.getenv(env_var), wrap_langsmith=wrap_langsmith)


def require_openai_client_from_env(
    env_var: str = "OPENAI_API_KEY",
    *,
    wrap_langsmith: bool = True,
) -> Any:
    client = get_openai_client_from_env(env_var, wrap_langsmith=wrap_langsmith)
    if client is None:
        raise RuntimeError(f"{env_var} is required.")
    return client