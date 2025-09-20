from typing import List, Optional

from pydantic import BaseModel, Field


class SpeciesMetadata(BaseModel):
    common_name: str
    scientific_name: str
    is_invasive: bool
    notes: Optional[str] = None


class ModelPrediction(BaseModel):
    label: str
    score: float
    species: Optional[SpeciesMetadata] = None


class ClassificationResponse(BaseModel):
    decision: str = Field(
        description="High-level summary string describing how the invasive flag was determined."
    )
    invasive: Optional[bool] = Field(
        default=None,
        description="Whether the system believes the organism is invasive (true), native (false), or unknown (None)."
    )
    threshold: float
    top_prediction: Optional[ModelPrediction] = None
    predictions: List[ModelPrediction] = Field(default_factory=list)
