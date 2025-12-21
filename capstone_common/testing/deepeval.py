from __future__ import annotations

import os
import pytest


def deepeval_pytestmark() -> pytest.MarkDecorator:
    """Skip DeepEval tests unless explicitly opted in (prevents surprise token spend)."""
    deepeval_run = os.getenv("DEEPEVAL_RUN", "0").lower() in ("1", "true", "yes", "y")
    openai_key = os.getenv("OPENAI_API_KEY")
    return pytest.mark.skipif(
        not (deepeval_run and openai_key),
        reason="DeepEval tests are opt-in. Set DEEPEVAL_RUN=1 and OPENAI_API_KEY.",
    )


def configure_deepeval_openai_env(default_model: str = "gpt-4o-mini") -> None:
    """
    Help DeepEval pick OpenAI as the judge model (safe defaults).
    Does nothing if OPENAI_API_KEY is not set.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return

    os.environ.setdefault("USE_OPENAI_MODEL", "1")
    os.environ.setdefault("OPENAI_MODEL_NAME", os.getenv("DEEPEVAL_OPENAI_MODEL", default_model))