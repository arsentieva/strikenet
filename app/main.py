from __future__ import annotations

from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from app.config import get_settings
from app.schemas import ClassificationResponse, ModelPrediction, SpeciesMetadata
from app.services.inference import InferenceError, classify_image
from app.data.species import lookup_species

app = FastAPI(title="StrikeNet Invasive Species Classifier")


@app.get("/health", tags=["system"])
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/classify", response_model=ClassificationResponse, tags=["classification"])
async def classify_species(image: UploadFile = File(...)) -> ClassificationResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only image uploads are supported."
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    settings = get_settings()

    try:
        predictions_raw = await classify_image(image_bytes)
    except InferenceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    predictions: List[ModelPrediction] = []

    for prediction in predictions_raw[: settings.top_k]:
        species = lookup_species(prediction.label)
        predictions.append(
            ModelPrediction(
                label=prediction.label,
                score=prediction.score,
                species=SpeciesMetadata(
                    common_name=species.common_name,
                    scientific_name=species.scientific_name,
                    is_invasive=species.is_invasive,
                    notes=species.notes or None,
                ) if species else None,
            )
        )

    threshold = settings.classification_confidence_threshold
    decision = "no-match"
    invasive: bool | None = None
    top_prediction = predictions[0] if predictions else None

    if top_prediction and top_prediction.species:
        species = top_prediction.species
        if top_prediction.score >= threshold:
            invasive = species.is_invasive
            decision = (
                "auto-flagged-invasive" if invasive else "auto-identified-native"
            )
        else:
            invasive = species.is_invasive if species.is_invasive else None
            decision = "low-confidence"
    elif top_prediction:
        decision = "unrecognized-species"

    if not predictions:
        decision = "no-predictions"

    return ClassificationResponse(
        decision=decision,
        invasive=invasive,
        threshold=threshold,
        top_prediction=top_prediction,
        predictions=predictions,
    )
