import os
import json
from typing import Optional
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

class GenerateSentenceResponse(BaseModel):
    sentence_with_blank: str
    target_word: str

class ValidateRequest(BaseModel):
    sentence_with_blank: str
    user_word: str

class ValidateResponse(BaseModel):
    is_correct: bool
    explanation: str

GENERATE_PROMPT = (
    "Generate a simple English sentence with one common noun or verb replaced by a blank '___'. "
    "The sentence should be easy to understand and have a clear missing part that could be filled by multiple logical words. "
    "Provide the response in JSON format with two keys: 'sentence_with_blank' and 'example_word'. "
    "Example: {\"sentence_with_blank\": \"The dog was chasing the ___.\", \"example_word\": \"CAT\"}. "
    "Make sure the example word is simple to spell (3-7 letters)."
)

VALIDATE_PROMPT = (
    "You are an English teacher. A user is playing a game where they fill in a blank in a sentence. "
    "Sentence: {sentence}\n"
    "User's word: {user_word}\n\n"
    "Check if the user's word makes sense grammatically and semantically in the context of the sentence. "
    "Be encouraging. If the word fits reasonably well, it's correct. "
    "Provide the response in JSON format with two keys: 'is_correct' (boolean) and 'explanation' (short string explaining why it fits or why it doesn't)."
)

@router.get("/sentence-game/generate", response_model=GenerateSentenceResponse)
def generate_sentence():
    model = _get_model()
    try:
        resp = model.generate_content(GENERATE_PROMPT)
        # Handle potential markdown in response
        text = resp.text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        data = json.loads(text)
        # Map example_word to target_word for frontend compatibility if needed, 
        # but the frontend will just treat it as an example.
        return GenerateSentenceResponse(
            sentence_with_blank=data["sentence_with_blank"],
            target_word=data["example_word"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Error: {str(e)}")

@router.post("/sentence-game/validate", response_model=ValidateResponse)
def validate_word(req: ValidateRequest):
    model = _get_model()
    try:
        prompt = VALIDATE_PROMPT.format(
            sentence=req.sentence_with_blank,
            user_word=req.user_word
        )
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        data = json.loads(text)
        return ValidateResponse(**data)
    except Exception as e:
        return ValidateResponse(
            is_correct=False,
            explanation="I couldn't quite check that. Try again!"
        )
