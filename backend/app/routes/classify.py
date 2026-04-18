from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.classifier import classify_landmarks

router = APIRouter()


class Landmark(BaseModel):
    x: float
    y: float
    z: float


class ClassifyRequest(BaseModel):
    landmarks: List[Landmark] = Field(..., min_length=21, max_length=21)


class ClassifyResponse(BaseModel):
    letter: str
    confidence: float


@router.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    points = [(lm.x, lm.y, lm.z) for lm in req.landmarks]
    try:
        letter, confidence = classify_landmarks(points)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return ClassifyResponse(letter=letter, confidence=confidence)
