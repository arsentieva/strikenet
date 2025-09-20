from typing import List, Optional

from pydantic import BaseModel, Field


class ModelPrediction(BaseModel):
    species: str
    score: float
    invasive: bool
    hunting_allowed: bool
    details: Optional[str] = None

