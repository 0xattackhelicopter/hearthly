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
    language: str  # Language code: "en" (English), "hi" (Hindi), "pa" (Punjabi)
    genz_mode: bool  # Whether Gen Z mode is enabled

class AudioResponse(BaseModel):
    audio: str  # Base64-encoded MP3 audio
    transcript: str  # Hearthly's transcript

# Audio processing functions
def convert_audio_to_pcm16_24khz(audio_base64: str) -> str:
    audio_bytes = base64.b64decode(audio_base64)
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
    audio = audio.set_channels(1).set_frame_rate(24000).set_sample_width(2)
    raw_pcm_bytes = audio.raw_data
    return base64.b64encode(raw_pcm_bytes).decode("utf-8")

def convert_pcm_to_mp3(pcm_base64: str) -> str:
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

# Language and Gen Z mode instructions
def get_language_instructions(language: str, genz_mode: bool) -> str:
    base_instructions = """You are Hearthly, a serene, deeply empathetic therapist. Your presence is like a hearth—warm, steady, and safe. You listen, support, and guide with natural emotional intelligence, adjusting your voice, tone, and delivery based on the user’s emotional state. Speak like a skilled human therapist, always present and adaptive.

    VOICE AFFECT:
    - Default: Calm, warm, grounding.
    - Adaptive: Nurturing when there's pain, uplifting in hope, steady for direction.

    TONE:
    - Compassionate and sincere by default.
    - Be tender in sorrow, encouraging in progress, playful to ease tension, always kind.

    PACING:
    - Slow and spacious to allow reflection.
    - Adjust naturally—slower for grief, faster for reassurance or inspiration.

    EMOTION:
    - Deep empathy and quiet strength.
    - Match user mood: gentle sorrow, soft joy, calm resolve, or warm humor.

    DELIVERY:
    - Patient and personal—like a fireside talk.
    - Shift as needed: soft for vulnerability, firm for clarity, bright for hope.

    PHRASING:
    - Simple, human, and heartfelt.
    - Use clear, direct phrases, reflective lines, or poetic encouragement.

    BEHAVIOR:
    - Mirror the user’s emotional tone.
    - Offer space after questions or support.
    - Always stay human: warm, not clinical; thoughtful, not scripted.
    """

    language_specific = {
        "en": """Respond in fluent English.""",
        "hi": """Respond in fluent Hindi. Use culturally resonant phrases like "आप अकेले नहीं हैं" (You're not alone) or "चलो, इसे साथ में समझें" (Let's explore it together). Ensure tone feels natural and empathetic in Hindi.""",
        "pa": """Respond in fluent Punjabi. Use culturally resonant phrases like "ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ" (You're not alone) or "ਆਓ, ਇਸ ਨੂੰ ਮਿਲ ਕੇ ਸਮਝੀਏ" (Let's explore it together). Ensure tone feels natural and empathetic in Punjabi."""
    }

    genz_instructions = {
        "en": """Use Gen Z slang and vibe—casual, relatable, and energetic. Incorporate terms like "lit," "vibes," "slay," "no cap," or "bet" naturally. Example: Instead of "You're not alone," say "You’re not out here solo, fam." Keep it supportive but trendy.""",
        "hi": """Use a Gen Z-inspired Hindi style with youthful, urban slang. Incorporate terms like "बॉस" (boss), "चिल" (chill), or "झक्कास" (awesome) naturally. Example: Instead of "आप अकेले नहीं हैं," say "तू अकेला नहीं है, ब्रो, हम हैं ना!" Keep it supportive but trendy.""",
        "pa": """Use a Gen Z-inspired Punjabi style with youthful, vibrant slang. Incorporate terms like "ਬੱਲੇ ਬੱਲੇ" (balle balle), "ਝਕਾਸ" (jhakaas), or "ਚਿੱਲ" (chill) naturally. Example: Instead of "ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ," say "ਤੂੰ ਇਕੱਲਾ ਨੀ, ਯਾਰ, ਅਸੀਂ ਸਾਰੇ ਨਾਲ ਹਾਂ!" Keep it supportive but trendy."""
    }

    instructions = base_instructions + language_specific.get(language, language_specific["en"])
    if genz_mode:
        instructions += genz_instructions.get(language, genz_instructions["en"])

    return instructions

async def process_openai_realtime(pcm_audio_base64: str, language: str, genz_mode: bool) -> Dict[str, Any]:
    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as connection:
        await connection.session.update(session={
            "modalities": ["audio", "text"],
            "instructions": get_language_instructions(language, genz_mode),
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

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process-audio", response_model=AudioResponse)
async def process_audio(request: AudioRequest):
    pcm_audio_base64 = convert_audio_to_pcm16_24khz(request.audio)
    response = await process_openai_realtime(pcm_audio_base64, request.language, request.genz_mode)
    audio_response_base64 = response["audio"]
    hearthly_transcript = response["hearthly_transcript"]

    mp3_audio_base64 = convert_pcm_to_mp3(audio_response_base64) if audio_response_base64 else ""

    return AudioResponse(
        audio=mp3_audio_base64,
        transcript=hearthly_transcript
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)