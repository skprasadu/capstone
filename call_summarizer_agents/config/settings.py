"""Application settings for API keys and model configuration."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Centralized configuration for external integrations."""

    openai_api_key: str | None = Field(
        default=None, description="API key used for OpenAI-powered summarization"
    )
    openai_model: str = Field(
        default="gpt-4o-mini", description="Model used for text summarization"
    )
    openai_temperature: float = Field(
        default=0.2, description="Sampling temperature for OpenAI requests"
    )

    whisper_api_key: str | None = Field(
        default=None, description="API key used for Whisper audio transcription"
    )
    whisper_model: str = Field(
        default="whisper-1", description="Model used for audio-to-text transcription"
    )

    class Config:
        env_file = ".env"
        env_prefix = ""


def load_settings() -> AppSettings:
    """Load application settings from environment variables or a .env file."""

    return AppSettings()

