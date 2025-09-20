"""Wrapper around external image classification using OpenAI vision models."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from typing import List
from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger("strikenet.inference")
logger.setLevel(logging.INFO)

_SYSTEM_PROMPT = (
    "You are a marine wildlife identification assistant. "
    "When provided with an image, respond ONLY with strict JSON matching this schema: "
    '{"predictions": [{"label": "lowercase common name", "scientific_name": "string", "confidence": 0.0-1.0}]}. '
    "Provide at most {top_k} predictions sorted by confidence descending. "
    "Use lowercase for the label field. If unsure, use label \"unknown\" and confidence 0.0."
)


class InferenceError(RuntimeError):
    """Raised when the upstream model call fails."""


@dataclass
class ModelPrediction:
    label: str
    score: float


async def classify_image(image_bytes: bytes, mime_type: str | None) -> List[ModelPrediction]:
    """Send the image to the configured OpenAI vision-capable model."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise InferenceError("OpenAI API key is not configured")

    client = OpenAI(api_key=settings.openai_api_key)
    image_base64 = base64.b64encode(image_bytes).decode("ascii")

    image_payload: dict[str, str] = {"data": image_base64}
    if mime_type:
        image_payload["mime_type"] = mime_type

    system_prompt = _SYSTEM_PROMPT.format(top_k=settings.top_k)

    try:
        response = await asyncio.to_thread(
            client.responses.create,
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Identify the species in this image."},
                        {"type": "input_image", "image": image_payload},
                    ],
                },
            ],
            temperature=settings.openai_temperature,
            max_output_tokens=settings.openai_max_output_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - we want to wrap any client errors
        logger.exception("OpenAI request failed")
        raise InferenceError(f"OpenAI request failed: {exc}") from exc

    try:
        text_output = response.output[0].content[0].text  # type: ignore[index]
    except (AttributeError, IndexError, KeyError) as exc:
        logger.exception("Unexpected response structure from OpenAI", extra={"response": response})
        raise InferenceError("Unexpected response structure from OpenAI") from exc

    logger.info("Received raw prediction payload", extra={"raw_output": text_output})

    try:
        payload = json.loads(text_output)
    except json.JSONDecodeError as exc:
        logger.exception("Failed to parse OpenAI response JSON", extra={"raw_output": text_output})
        raise InferenceError("Failed to parse OpenAI response JSON") from exc

    raw_predictions = payload.get("predictions") if isinstance(payload, dict) else None
    if not isinstance(raw_predictions, list):
        raise InferenceError("OpenAI response missing predictions list")

    predictions: List[ModelPrediction] = []
    for entry in raw_predictions:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        confidence = entry.get("confidence")
        if label is None or confidence is None:
            continue
        try:
            predictions.append(ModelPrediction(label=str(label), score=float(confidence)))
        except (TypeError, ValueError):
            continue

    logger.info(
        "Parsed predictions",
        extra={
            "prediction_count": len(predictions),
            "top_label": predictions[0].label if predictions else None,
            "top_score": predictions[0].score if predictions else None,
        },
    )

    return predictions
