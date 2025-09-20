from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables with prefix STRIKENET_."""

    huggingface_api_token: Optional[str] = Field(
        default=None,
        description="Bearer token for Hugging Face Inference API access."
    )
    huggingface_model_id: str = Field(
        default="microsoft/resnet-50",
        description="Default image classification model deployed on Hugging Face."
    )
    huggingface_api_url: Optional[str] = Field(
        default=None,
        description="Override full Hugging Face Inference API URL if using a custom endpoint."
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

    @property
    def resolved_huggingface_url(self) -> str:
        if self.huggingface_api_url:
            return self.huggingface_api_url
        return f"https://api-inference.huggingface.co/models/{self.huggingface_model_id}"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
