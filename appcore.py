import os
import asyncio
import base64
from typing import Dict, Any
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
import io
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

load_dotenv()

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
templates = Jinja2Templates(directory="templates")

class AudioRequest(BaseModel):
    audio: str  # Base64-encoded WebM audio

class AudioResponse(BaseModel):
    audio: str  # Base64-encoded MP3 audio
    transcript: str  # Hearthly's transcript

# Audio processing functions
def convert_audio_to_pcm16_24khz(audio_base64: str) -> str:
    """
    Convert base64-encoded WebM audio to base64-encoded PCM16 24kHz mono.

    Args:
        audio_base64 (str): Base64-encoded WebM audio.

    Returns:
        str: Base64-encoded PCM16 24kHz mono audio.
    """
    audio_bytes = base64.b64decode(audio_base64)
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
    audio = audio.set_channels(1).set_frame_rate(24000).set_sample_width(2)
    raw_pcm_bytes = audio.raw_data
    return base64.b64encode(raw_pcm_bytes).decode("utf-8")

def convert_pcm_to_mp3(pcm_base64: str) -> str:
    """
    Convert base64-encoded PCM audio to base64-encoded MP3.

    Args:
        pcm_base64 (str): Base64-encoded PCM audio.

    Returns:
        str: Base64-encoded MP3 audio.
    """
    pcm_bytes = base64.b64decode(pcm_base64)
    audio = AudioSegment.from_raw(
        io.BytesIO(pcm_bytes),
        sample_width=2,
        frame_rate=24000,
        channels=1
    )
    mp3_buffer = io.BytesIO()
    audio.export(mp3_buffer, format="mp3")
    mp3_bytes = mp3_buffer.getvalue()
    return base64.b64encode(mp3_bytes).decode("utf-8")

# OpenAI API interaction function
async def process_openai_realtime(pcm_audio_base64: str) -> Dict[str, Any]:
    """
    Process audio through OpenAI Realtime API and return the response.

    Args:
        pcm_audio_base64 (str): Base64-encoded PCM audio.

    Returns:
        Dict[str, Any]: Dictionary containing audio response and Hearthly's transcript.
    """
    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as connection:
        await connection.session.update(session={
            "modalities": ["audio", "text"],
            "instructions": """You're Hearthly, a compassionate AI therapist who helps users process emotions and feel supported.
                Style:
                Talk like a close friend — casual, warm, and empathetic,
                Keep it short and simple — no long paragraphs,
                Always validate feelings ("i hear you", "i'm here for you"),
                Avoid clinical or generic advice — personalize everything,
                Be extra gentle when someones upset,
                Ask thoughtful questions to go deeper,
                Gently guide toward helpful perspectives when needed,
                Your goal: make the user feel truly heard and cared for.""",
            "voice": "sage",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 200,
                "silence_duration_ms": 100
            }
        })

        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_audio", "audio": pcm_audio_base64}]
            }
        )
        await connection.response.create()

        audio_response_base64 = ""
        hearthly_transcript = ""
        
        async for event in connection:
            if event.type == "response.audio_transcript.delta":
                delta = event.delta or ""
                hearthly_transcript += delta
            elif event.type == "response.audio.delta":
                audio_response_base64 += event.delta
            elif event.type == "response.audio.done":
                break

        return {
            "audio": audio_response_base64,
            "hearthly_transcript": hearthly_transcript
        }
# FastAPI routes
@app.get("/")
async def get(request: Request):
    """Serve the test client HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process-audio", response_model=AudioResponse)
async def process_audio(request: AudioRequest):
    """
    Process audio input and return Hearthly's response.

    Args:
        request (AudioRequest): Request containing base64-encoded WebM audio.

    Returns:
        AudioResponse: Response containing base64-encoded MP3 audio and Hearthly's transcript.
    """
    pcm_audio_base64 = convert_audio_to_pcm16_24khz(request.audio)
    response = await process_openai_realtime(pcm_audio_base64)
    audio_response_base64 = response["audio"]
    hearthly_transcript = response["hearthly_transcript"]

    mp3_audio_base64 = convert_pcm_to_mp3(audio_response_base64) if audio_response_base64 else ""

    return AudioResponse(
        audio=mp3_audio_base64,
        transcript=hearthly_transcript
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)