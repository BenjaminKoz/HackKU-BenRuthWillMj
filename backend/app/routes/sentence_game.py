import random
from typing import List
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# Preloaded dataset of sentences and valid answers
SENTENCES = [
    {"s": "The dog was chasing the ___.", "v": ["CAT", "BALL", "SQUIRREL", "CAR", "BIRD", "RABBIT"]},
    {"s": "I like to eat a red ___.", "v": ["APPLE", "CHERRY", "BERRY", "TOMATO", "FRUIT"]},
    {"s": "The sun is a hot ___.", "v": ["STAR", "BALL", "LIGHT", "FIRE", "CORE"]},
    {"s": "Please open the ___.", "v": ["DOOR", "WINDOW", "BOX", "BOOK", "JAR", "CAN"]},
    {"s": "The blue ___ is in the sky.", "v": ["BIRD", "PLANE", "KITE", "CLOUD", "MOON"]},
    {"s": "I drink water from a ___.", "v": ["CUP", "GLASS", "MUG", "BOTTLE", "BOWL"]},
    {"s": "The cat sat on the ___.", "v": ["MAT", "RUG", "COUCH", "CHAIR", "TABLE", "FLOOR", "LAP"]},
    {"s": "She wears a hat on her ___.", "v": ["HEAD", "HAIR", "FACE"]},
    {"s": "I write with a pen on ___.", "v": ["PAPER", "BOOK", "DESK", "PAGE"]},
    {"s": "The fish swims in the ___.", "v": ["WATER", "LAKE", "SEA", "OCEAN", "POND", "TANK"]},
    {"s": "A big elephant has a long ___.", "v": ["TRUNK", "NOSE", "TAIL"]},
    {"s": "He drives a fast red ___.", "v": ["CAR", "TRUCK", "VAN", "BIKE", "MOTOR"]},
    {"s": "The tree has green ___.", "v": ["LEAVES", "LEAF", "FRUIT", "BRANCH", "WOOD"]},
    {"s": "I sleep in a warm ___.", "v": ["BED", "ROOM", "HOUSE", "BAG", "TENT"]},
    {"s": "The rain falls from the ___.", "v": ["SKY", "CLOUD", "TOP"]},
    {"s": "You use a key to open a ___.", "v": ["LOCK", "DOOR", "GATE", "SAFE", "BOX"]},
    {"s": "A monkey likes to eat a ___.", "v": ["BANANA", "FRUIT", "NUT", "BERRY"]},
    {"s": "I can see with my two ___.", "v": ["EYES", "SIGHT", "LENS"]},
    {"s": "The baker makes fresh ___.", "v": ["BREAD", "CAKE", "PIE", "FOOD", "ROLLS"]},
    {"s": "We play music on a ___.", "v": ["PIANO", "DRUM", "GUITAR", "FLUTE", "HORN", "STAGE"]}
]

class GenerateSentenceResponse(BaseModel):
    sentence_with_blank: str
    target_word: str
    possible_words: List[str]

class ValidateRequest(BaseModel):
    sentence_with_blank: str
    user_word: str

class ValidateResponse(BaseModel):
    is_correct: bool
    explanation: str

@router.get("/sentence-game/generate", response_model=GenerateSentenceResponse)
def generate_sentence():
    item = random.choice(SENTENCES)
    possible_words = item["v"][:]
    random.shuffle(possible_words)
    return GenerateSentenceResponse(
        sentence_with_blank=item["s"],
        target_word=item["v"][0],
        possible_words=possible_words,
    )

@router.post("/sentence-game/validate", response_model=ValidateResponse)
def validate_word(req: ValidateRequest):
    user_word = req.user_word.upper().strip()
    
    # Find the sentence in our dataset
    valid_words = []
    for item in SENTENCES:
        if item["s"] == req.sentence_with_blank:
            valid_words = item["v"]
            break
    
    if not valid_words:
        # Fallback if sentence wasn't found (shouldn't happen)
        is_correct = len(user_word) >= 3
        return ValidateResponse(
            is_correct=is_correct,
            explanation="That fits!" if is_correct else "Try a longer word."
        )

    if user_word in valid_words:
        return ValidateResponse(
            is_correct=True,
            explanation=f"Yes! '{user_word}' fits perfectly."
        )
    else:
        example = random.choice(valid_words)
        return ValidateResponse(
            is_correct=False,
            explanation=f"Not quite. A word like '{example}' would fit well here."
        )
