import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from elevenlabs.client import ElevenLabs

router = APIRouter()

_client = None


def _get_client() -> ElevenLabs:
    global _client
    if _client is None:
        key = os.getenv("ELEVENLABS_API_KEY")
        if not key:
            raise HTTPException(status_code=503, detail="ELEVENLABS_API_KEY not set")
        _client = ElevenLabs(api_key=key)
    return _client


class SpeakRequest(BaseModel):
    text: str


@router.post("/speak")
def speak(req: SpeakRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    client = _get_client()
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

    audio_iter = client.text_to_speech.convert(
        voice_id=voice_id,
        text=req.text,
        model_id="eleven_flash_v2_5",
        output_format="mp3_44100_128",
    )

    def stream():
        for chunk in audio_iter:
            if chunk:
                yield chunk

    return StreamingResponse(stream(), media_type="audio/mpeg")
