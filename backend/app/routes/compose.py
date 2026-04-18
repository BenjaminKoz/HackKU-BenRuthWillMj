import os
from typing import Literal
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
        _model = genai.GenerativeModel("gemini-1.5-flash")
    return _model


class ComposeRequest(BaseModel):
    letters: str
    mode: Literal["letters", "words"] = "letters"


class ComposeResponse(BaseModel):
    text: str


LETTERS_PROMPT = (
    "You translate streams of ASL alphabet letters detected from a webcam into "
    "a natural English sentence. The input may contain repeated letters (held "
    "signs), gaps, or misreads. Infer the most likely intended sentence. "
    "Respond with ONLY the sentence, no preamble, no quotes."
)

WORDS_PROMPT = (
    "You turn a sequence of ASL word glosses recognized from a webcam into a "
    "natural English sentence. Input is uppercase words separated by spaces "
    "(e.g. 'HELLO YOU PLEASE HELP'). Produce a short, natural sentence that "
    "expresses the same meaning with correct English grammar, pronouns, and "
    "articles. Respond with ONLY the sentence, no preamble, no quotes."
)


@router.post("/compose", response_model=ComposeResponse)
def compose(req: ComposeRequest) -> ComposeResponse:
    if not req.letters.strip():
        return ComposeResponse(text="")
    model = _get_model()
    system_prompt = WORDS_PROMPT if req.mode == "words" else LETTERS_PROMPT
    label = "Words" if req.mode == "words" else "Letters"
    prompt = f"{system_prompt}\n\n{label}: {req.letters}"
    resp = model.generate_content(prompt)
    return ComposeResponse(text=(resp.text or "").strip())
