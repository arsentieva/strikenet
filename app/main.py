from __future__ import annotations

import logging
from fastapi import FastAPI, File, HTTPException, UploadFile, status

from app.services.inference import InferenceError, classify_image

logger = logging.getLogger("strikenet.api")
logger.setLevel(logging.INFO)

app = FastAPI(title="StrikeNet Invasive Species Classifier")


@app.get("/health", tags=["system"])
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/classify", tags=["classification"])
async def classify_species(image: UploadFile = File(...)):
    logger.info("Received classification request", extra={"content_type": image.content_type})

    if not image.content_type or not image.content_type.startswith("image/"):
        logger.warning("Rejected upload with unsupported content type", extra={"content_type": image.content_type})
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only image uploads are supported."
        )

    image_bytes = await image.read()
    if not image_bytes:
        logger.warning("Rejected upload with empty payload")
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info("Read image payload", extra={"size_bytes": len(image_bytes)})
   

    try:
        return await classify_image(image_bytes, image.content_type)
    except InferenceError as exc:
        logger.exception("Inference call failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
