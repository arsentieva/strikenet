"""Wrapper around external image classification using OpenAI vision models."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("STRIKENET_OPENAI_API_KEY")
logger = logging.getLogger("strikenet.inference")
logger.setLevel(logging.INFO)

_SYSTEM_PROMPT = (
    "You are a wildlife identification assistant who specializes in identifying species from images and providing detailed information about"
    "them and clarify if they are they invasive to south florida and allowed to be hunted. "
    "When provided with an image, respond  JSON"
    "Use lowercase for the label field. If unsure, use label \"unknown\" and confidence 0.0."
)


class InferenceError(RuntimeError):
    """Raised when the upstream model call fails."""


@dataclass
class ModelPrediction:
    label: str
    score: float


async def classify_image(image_bytes: bytes, mime_type: str | None) -> Any:
    """Send the image to the configured OpenAI vision-capable model."""

    # client = OpenAI(api_key=API_KEY)
    # image_base64 = base64.b64encode(image_bytes).decode("ascii")

    # data_uri_mime = mime_type or "image/png"
    # image_data_uri = f"data:{data_uri_mime};base64,{image_base64}"

    # system_prompt = _SYSTEM_PROMPT.format(top_k=os.getenv("STRIKENET_TOP_K", 5))

    # try:
    #     response = await asyncio.to_thread(
    #         client.responses.create,
    #         model=os.getenv("STRIKENET_OPENAI_MODEL", "gpt-4o-mini"),
    #         input=[
    #             {
    #                 "role": "system",
    #                 "content": [
    #                     {
    #                         "type": "input_text",
    #                         "text": system_prompt,
    #                     }
    #                 ],
    #             },
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "input_text", "text": "Identify the species in this image."},
    #                     {"type": "input_image", "image_url": image_data_uri},
    #                 ],
    #             },
    #         ],
    #         temperature=0.0,
    #         max_output_tokens=600,
    #     )
    # except Exception as exc:  # noqa: BLE001 - we want to wrap any client errors
    #     logger.exception("OpenAI request failed")
    #     raise InferenceError(f"OpenAI request failed: {exc}") from exc

    response = "```json\n{\n  \"label\": \"peacock\",\n \"confidence\": 0.95,\n  \"invasive\": false,\n  \"hunting_allowed\": false,\n  \"details\": {\n    \"scientific_name\": \"Pavo cristatus\",\n    \"description\": \"Peacocks are large, colorful birds known for their iridescent tail feathers, which they fan out during courtship displays. They are native to South Asia but have been introduced to various regions worldwide.\",\n    \"habitat\": \"Peacocks prefer open forests, grasslands, and areas near water.\",\n    \"behavior\": \"They are omnivorous, feeding on seeds, insects, and small animals.\"\n  }\n}\n```"

    def parse_response(response: Any) -> any:
        try:
            parsed_response = json.loads(response.split("```json")[-1].split("```")[0].strip())

            return {"species": parsed_response["label"],
                    "score": float(parsed_response["confidence"]),
                    "invasive": bool(parsed_response["invasive"]),
                    "hunting_allowed": bool(parsed_response["hunting_allowed"]),
                    "details": parsed_response["details"]}

        except (KeyError, json.JSONDecodeError) as exc:
            logger.exception("Failed to parse model response")
            raise InferenceError(f"Failed to parse model response: {exc}") from exc

    return parse_response(response)


