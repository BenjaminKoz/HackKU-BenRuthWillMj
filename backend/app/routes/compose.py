import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import google.generativeai as genai

router = APIRouter()

_model = None


def _get_model():
    global _model
    if _model is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise HTTPException(status_code=503, detail="GEMINI_API_KEY not set")
        genai.configure(api_key=key)
        _model = genai.GenerativeModel("gemini-2.5-flash")
    return _model


class ComposeRequest(BaseModel):
    letters: str


class ComposeResponse(BaseModel):
    text: str


SYSTEM_PROMPT = (
    "You translate streams of ASL alphabet letters detected from a webcam into "
    "a natural English sentence. The input may contain repeated letters (held "
    "signs), gaps, or misreads. Infer the most likely intended sentence. "
    "Respond with ONLY the sentence, no preamble, no quotes."
)


@router.post("/compose", response_model=ComposeResponse)
def compose(req: ComposeRequest) -> ComposeResponse:
    if not req.letters.strip():
        return ComposeResponse(text="")
    model = _get_model()
    prompt = f"{SYSTEM_PROMPT}\n\nLetters: {req.letters}"
    resp = model.generate_content(prompt)
    return ComposeResponse(text=(resp.text or "").strip())
