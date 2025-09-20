from functools import lru_cache

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables with prefix STRIKENET_."""

    openai_api_key: str = Field(
        ..., description="API key for accessing OpenAI services."
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model identifier used for vision classification."
    )
    openai_temperature: float = Field(
        default=0.0,
        description="Sampling temperature passed to the OpenAI model."
    )
    openai_max_output_tokens: int = Field(
        default=600,
        description="Maximum number of output tokens requested from the OpenAI model."
    )
    top_k: int = Field(
        default=5,
        description="Number of predictions to request from the upstream model."
    )
    classification_confidence_threshold: float = Field(
        default=0.6,
        description="Confidence threshold needed to auto-flag an invasive species."
    )

    class Config:
        env_prefix = "STRIKENET_"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
