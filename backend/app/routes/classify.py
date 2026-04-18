from typing import List, Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.classifier import (
    WORD_CLIP_FRAMES,
    classify_landmarks,
    classify_word_clip,
    get_word_labels,
    reset_smoothing,
)

router = APIRouter()


class Landmark(BaseModel):
    x: float
    y: float
    z: float


class ClassifyRequest(BaseModel):
    landmarks: List[Landmark] = Field(..., min_length=21, max_length=21)
    mode: Literal["letters", "words"] = "letters"


class ClassifyResponse(BaseModel):
    letter: str
    confidence: float


class ClassifyWordClipRequest(BaseModel):
    frames: List[List[List[Landmark]]] = Field(
        ...,
        description=(
            "Exactly WORD_CLIP_FRAMES frames. Each frame is a list of 0, 1, "
            "or 2 detected hands (each with 21 landmarks). Should be captured "
            "at ~30Hz to match training data."
        ),
    )


@router.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    points = [(lm.x, lm.y, lm.z) for lm in req.landmarks]
    try:
        letter, confidence = classify_landmarks(points, mode=req.mode)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return ClassifyResponse(letter=letter, confidence=confidence)


@router.post("/classify-word-clip", response_model=ClassifyResponse)
def classify_word_clip_route(req: ClassifyWordClipRequest) -> ClassifyResponse:
    if len(req.frames) != WORD_CLIP_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {WORD_CLIP_FRAMES} frames, got {len(req.frames)}.",
        )
    for i, frame in enumerate(req.frames):
        if len(frame) > 2:
            raise HTTPException(
                status_code=400,
                detail=f"Frame {i}: got {len(frame)} hands, max is 2.",
            )
        for j, hand in enumerate(frame):
            if len(hand) != 21:
                raise HTTPException(
                    status_code=400,
                    detail=f"Frame {i} hand {j}: expected 21 landmarks, got {len(hand)}.",
                )
    frames = [
        [[(lm.x, lm.y, lm.z) for lm in hand] for hand in frame]
        for frame in req.frames
    ]
    try:
        letter, confidence = classify_word_clip(frames)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return ClassifyResponse(letter=letter, confidence=confidence)


@router.post("/classify/reset")
def classify_reset() -> dict:
    """Frontend calls this when the hand leaves the frame so the smoothing
    buffer doesn't blend the previous sign into the next one."""
    reset_smoothing()
    return {"ok": True}


@router.get("/words")
def list_words() -> dict:
    """Return the labels the current word model was trained on.

    The frontend uses this to render the "words you can sign" chip list and
    to generate challenge sentences — so the UI stays in sync with whatever
    vocabulary was last trained, without hardcoding it."""
    return {"words": get_word_labels()}
