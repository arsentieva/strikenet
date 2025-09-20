"""Wrapper around external image classification models."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

import httpx

from app.config import get_settings


class InferenceError(RuntimeError):
    """Raised when the upstream model call fails."""


@dataclass
class ModelPrediction:
    label: str
    score: float


async def classify_image(image_bytes: bytes) -> List[ModelPrediction]:
    """Send the image to the configured Hugging Face inference endpoint."""
    settings = get_settings()
    headers = {"Accept": "application/json"}
    if settings.huggingface_api_token:
        headers["Authorization"] = f"Bearer {settings.huggingface_api_token}"
    # Hugging Face expects raw binary body for image classification.
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            settings.resolved_huggingface_url,
            headers=headers,
            content=image_bytes,
        )
    if response.status_code == 429:
        raise InferenceError("Upstream model is throttling requests (HTTP 429)")
    if response.status_code >= 400:
        raise InferenceError(
            f"Model request failed with status {response.status_code}: {response.text}"
        )

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise InferenceError("Failed to decode inference response as JSON") from exc

    # Hugging Face responses can be nested; normalize to flat list.
    if isinstance(payload, list):
        raw_predictions = payload
    elif isinstance(payload, dict) and "label" in payload:
        raw_predictions = [payload]
    elif isinstance(payload, dict):
        raw_predictions = payload.get("outputs") or payload.get("data") or []
    else:
        raw_predictions = []

    predictions: List[ModelPrediction] = []
    for entry in raw_predictions:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label") or entry.get("class")
        score = entry.get("score") or entry.get("probability")
        if label is None or score is None:
            continue
        predictions.append(ModelPrediction(label=str(label), score=float(score)))

    return predictions
