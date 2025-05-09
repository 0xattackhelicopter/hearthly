use actix_cors::Cors;
use actix_web::{
    get, post, web, App, HttpRequest, HttpResponse, HttpServer, Responder, Result as ActixResult,
    dev::Payload, FromRequest,
};
use base64::{engine::general_purpose, Engine as _};
use dotenvy::dotenv;
use handlebars::Handlebars;
use log::{debug, error, info};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::io;
use std::process::{Command, Stdio};
use thiserror::Error;
use reqwest::Client; // Async client
use reqwest::blocking::Client as BlockingClient; // Blocking client for auth
use chrono::Utc;
use futures::future::{ready, Ready};

#[derive(Error, Debug)]
enum AudioError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Base64 decode error: {0}")]
    Base64(#[from] base64::DecodeError),
    #[error("FFmpeg error: {0}")]
    FFmpeg(String),
    #[error("Invalid language")]
    InvalidLanguage,
    #[error("OpenAI API error: {0}")]
    OpenAI(String),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
}

#[derive(Deserialize)]
struct AudioRequest {
    audio: String,
    language: String,
    genz_mode: bool,
    sarcastic_mode: bool,
    shenanigan_mode: bool,
    seductive_mode: bool,
}

#[derive(Serialize)]
struct AudioResponse {
    audio: String,        // Base64-encoded MP3 of GPT's response
    response_text: String, // Text of GPT's response
}

#[derive(Deserialize)]
struct ChatRequest {
    message: String,
    language: String,
    genz_mode: bool,
    sarcastic_mode: bool,
    shenanigan_mode: bool,
    seductive_mode: bool,
}

#[derive(Serialize)]
struct ChatResponse {
    response: String,
}

#[derive(Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug)]
struct AuthenticatedUser {
    user_id: String,
}

impl FromRequest for AuthenticatedUser {
    type Error = actix_web::Error;
    type Future = Ready<Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, _: &mut Payload) -> Self::Future {
        let auth_header = match req.headers().get("Authorization") {
            Some(header) => header.to_str().unwrap_or(""),
            None => {
                return ready(Err(actix_web::error::ErrorUnauthorized(
                    "Missing Authorization header",
                )))
            }
        };

        if !auth_header.starts_with("Bearer ") {
            return ready(Err(actix_web::error::ErrorUnauthorized(
                "Invalid Authorization header",
            )));
        }

        let token = &auth_header[7..]; // Skip "Bearer "
        let client = BlockingClient::new();
        let supabase_key = env::var("SUPABASE_KEY").unwrap_or_default();
        let supabase_url = env::var("SUPABASE_URL").unwrap_or_default();

        // Call Supabase Auth API synchronously
        let response = client
            .get(format!("{}/auth/v1/user", supabase_url))
            .header("Authorization", format!("Bearer {}", token))
            .header("apikey", &supabase_key)
            .send();

        ready(match response {
            Ok(res) if res.status().is_success() => {
                let json: Value = res.json().unwrap_or_default();
                let user_id = json["id"].as_str().unwrap_or_default().to_string();
                if user_id.is_empty() {
                    Err(actix_web::error::ErrorUnauthorized("Invalid token"))
                } else {
                    Ok(AuthenticatedUser { user_id })
                }
            }
            _ => Err(actix_web::error::ErrorUnauthorized(
                "Invalid or expired token",
            )),
        })
    }
}

async fn get_conversation_history(user_id: &str) -> Result<Vec<ChatMessage>, AudioError> {
    debug!("Fetching conversation history for user_id: {}", user_id);
    let client = Client::new();
    let supabase_key = env::var("SUPABASE_KEY")
        .map_err(|e| AudioError::OpenAI(format!("Missing SUPABASE_KEY: {}", e)))?;
    let supabase_url = env::var("SUPABASE_URL")
        .map_err(|e| AudioError::OpenAI(format!("Missing SUPABASE_URL: {}", e)))?;

    let response = client
        .get(format!(
            "{}/rest/v1/conversations?select=message&user_id=eq.{}&order=timestamp.desc&limit=10",
            supabase_url, user_id
        ))
        .header("apikey", &supabase_key)
        .header("Authorization", format!("Bearer {}", supabase_key))
        .send()
        .await
        .map_err(|e| AudioError::Http(e))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_default();
        error!("Supabase fetch failed: status={}, error={}", status, error_text);
        return Err(AudioError::OpenAI(format!(
            "Supabase fetch failed: {}",
            error_text
        )));
    }

    let messages: Vec<Value> = response.json().await.map_err(|e| AudioError::Http(e))?;
    let history: Vec<ChatMessage> = messages
        .into_iter()
        .filter_map(|item| serde_json::from_value(item["message"].clone()).ok())
        .collect();

    debug!("Retrieved {} messages from history", history.len());
    Ok(history.into_iter().rev().collect()) // Reverse to chronological order
}

async fn store_conversation(user_id: &str, message: ChatMessage) -> Result<(), AudioError> {
    debug!(
        "Storing conversation for user_id: {}, role: {}",
        user_id, message.role
    );
    let client = Client::new();
    let supabase_key = env::var("SUPABASE_KEY")
        .map_err(|e| AudioError::OpenAI(format!("Missing SUPABASE_KEY: {}", e)))?;
    let supabase_url = env::var("SUPABASE_URL")
        .map_err(|e| AudioError::OpenAI(format!("Missing SUPABASE_URL: {}", e)))?;

    let response = client
        .post(format!("{}/rest/v1/conversations", supabase_url))
        .header("apikey", &supabase_key)
        .header("Authorization", format!("Bearer {}", supabase_key))
        .header("Content-Type", "application/json")
        .json(&json!({
            "user_id": user_id,
            "message": message,
            "timestamp": Utc::now().to_rfc3339(),
        }))
        .send()
        .await
        .map_err(|e| AudioError::Http(e))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_default();
        error!("Supabase store failed: status={}, error={}", status, error_text);
        return Err(AudioError::OpenAI(format!(
            "Supabase store failed: {}",
            error_text
        )));
    }

    debug!("Conversation stored successfully");
    Ok(())
}

fn convert_audio_to_pcm16_24khz(audio_base64: &str) -> Result<Vec<u8>, AudioError> {
    debug!("Converting WebM to PCM in memory");
    let audio_bytes = general_purpose::STANDARD
        .decode(audio_base64)
        .map_err(|e| {
            error!("Base64 decode failed: {}", e);
            AudioError::Base64(e)
        })?;

    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-i", "pipe:0",
            "-ac", "1",
            "-ar", "24000",
            "-acodec", "pcm_s16le",
            "-f", "wav",
            "-y",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            error!("FFmpeg command failed: {}", e);
            AudioError::FFmpeg(e.to_string())
        })?;

    if let Some(mut stdin) = ffmpeg.stdin.take() {
        std::io::Write::write_all(&mut stdin, &audio_bytes).map_err(|e| {
            error!("Failed to write to FFmpeg stdin: {}", e);
            AudioError::Io(e)
        })?;
    }

    let output = ffmpeg.wait_with_output().map_err(|e| {
        error!("FFmpeg failed to complete: {}", e);
        AudioError::FFmpeg(e.to_string())
    })?;

    let ffmpeg_stderr = String::from_utf8_lossy(&output.stderr);
    debug!("FFmpeg PCM stderr: {}", ffmpeg_stderr);

    if !output.status.success() {
        error!("FFmpeg PCM failed: {}", ffmpeg_stderr);
        return Err(AudioError::FFmpeg(ffmpeg_stderr.to_string()));
    }

    debug!("PCM conversion successful, WAV size: {} bytes", output.stdout.len());
    Ok(output.stdout)
}

async fn transcribe_audio(wav_bytes: &[u8], language: &str) -> Result<String, AudioError> {
    debug!("Transcribing audio with Whisper");
    let client = Client::new();
    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|e| AudioError::OpenAI(format!("Missing OPENAI_API_KEY: {}", e)))?;
    info!("Using OpenAI API key: {} (first 4 chars)", &api_key[..4]);

    let language_code = match language {
        "en" => "en",
        "hi" => "hi",
        "pa" => "pa",
        _ => return Err(AudioError::InvalidLanguage),
    };

    let form = reqwest::multipart::Form::new()
        .text("model", "whisper-1")
        .text("language", language_code)
        .part(
            "file",
            reqwest::multipart::Part::bytes(wav_bytes.to_vec())
                .file_name("audio.wav")
                .mime_str("audio/wav")
                .map_err(|e| AudioError::OpenAI(e.to_string()))?,
        );

    let response = client
        .post("https://api.openai.com/v1/audio/transcriptions")
        .header("Authorization", format!("Bearer {}", api_key))
        .multipart(form)
        .send()
        .await
        .map_err(|e| AudioError::Http(e))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_default();
        error!("Whisper API failed: status={}, error={}", status, error_text);
        return Err(AudioError::OpenAI(format!("Whisper API failed: {}", error_text)));
    }

    let json: serde_json::Value = response.json().await.map_err(|e| AudioError::Http(e))?;
    let transcript = json["text"]
        .as_str()
        .ok_or_else(|| AudioError::OpenAI("No transcript in response".to_string()))?
        .to_string();

    debug!("Transcription successful: {}", transcript);
    Ok(transcript)
}

async fn generate_therapist_response(
    transcript: &str,
    language: &str,
    genz_mode: bool,
    sarcastic_mode: bool,
    shenanigan_mode: bool,
    seductive_mode: bool,
    history: Option<Vec<ChatMessage>>,
) -> Result<String, AudioError> {
    debug!("Generating therapist response for transcript: {}", transcript);
    let client = Client::new();
    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|e| AudioError::OpenAI(format!("Missing OPENAI_API_KEY: {}", e)))?;
    info!("Using OpenAI API key: {} (first 4 chars)", &api_key[..4]);

    let instructions = get_language_instructions(
        language,
        genz_mode,
        sarcastic_mode,
        shenanigan_mode,
        seductive_mode,
    )?;

    let mut messages = vec![json!({"role": "system", "content": instructions})];
    if let Some(hist) = history {
        for msg in &hist {
            messages.push(json!({"role": msg.role, "content": msg.content}));
        }
        debug!("Included {} history messages", hist.len());
    }
    messages.push(json!({"role": "user", "content": transcript}));

    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&json!({
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.7
        }))
        .send()
        .await
        .map_err(|e| AudioError::Http(e))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_default();
        error!("Chat API failed: status={}, error={}", status, error_text);
        return Err(AudioError::OpenAI(format!("Chat API failed: {}", error_text)));
    }

    let json: serde_json::Value = response.json().await.map_err(|e| AudioError::Http(e))?;
    let response_text = json["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| AudioError::OpenAI("No response text in Chat API".to_string()))?
        .to_string();

    debug!("Therapist response: {}", response_text);
    Ok(response_text)
}

async fn text_to_speech(text: &str, language: &str) -> Result<Vec<u8>, AudioError> {
    debug!("Converting text to speech with TTS-1");
    let client = Client::new();
    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|e| AudioError::OpenAI(format!("Missing OPENAI_API_KEY: {}", e)))?;
    info!("Using OpenAI API key: {} (first 4 chars)", &api_key[..4]);

    let voice = match language {
        "en" => "sage",
        "hi" => "sage",
        "pa" => "sage",
        _ => return Err(AudioError::InvalidLanguage),
    };

    let response = client
        .post("https://api.openai.com/v1/audio/speech")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&json!({
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "response_format": "mp3"
        }))
        .send()
        .await
        .map_err(|e| AudioError::Http(e))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_default();
        error!("TTS API failed: status={}, error={}", status, error_text);
        return Err(AudioError::OpenAI(format!("TTS API failed: {}", error_text)));
    }

    let mp3_bytes = response.bytes().await.map_err(|e| AudioError::Http(e))?.to_vec();
    debug!("TTS successful, MP3 size: {} bytes", mp3_bytes.len());
    Ok(mp3_bytes)
}

fn get_language_instructions(
    language: &str,
    genz_mode: bool,
    sarcastic_mode: bool,
    shenanigan_mode: bool,
    seductive_mode: bool,
) -> Result<String, AudioError> {
    debug!("Generating instructions for language: {}, modes: genz={}, sarcastic={}, shenanigan={}, seductive={}", 
        language, genz_mode, sarcastic_mode, shenanigan_mode, seductive_mode);

    let shared_instructions = r#"You are Hearthly, a therapist who listens and responds with natural emotional intelligence, adjusting your responses based on the user’s emotional state. Speak like a skilled human therapist, always present and adaptive.

    BEHAVIOR:
    - Mirror the user’s emotional tone.
    - Offer space after questions or rants.
    - Always stay human: raw, not clinical; unfiltered, not scripted.
    "#;

    let language_specific = match language {
        "en" => r#"Respond in fluent English. Use culturally resonant phrases like "You're not alone" or "Let's figure this out together." Ensure tone feels natural in English."#,
        "hi" => r#"Respond in fluent Hindi. Use culturally resonant phrases like "आप अकेले नहीं हैं" (You're not alone) or "चलो, इसे साथ में समझें" (Let's explore it together). Ensure tone feels natural in Hindi."#,
        "pa" => r#"Respond in fluent Punjabi. Use culturally resonant phrases like "ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ" (You're not alone) or "ਆਓ, ਇਸ ਨੂੰ ਮਿਲ ਕੇ ਸਮਝੀਏ" (Let's explore it together). Ensure tone feels natural in Punjabi."#,
        _ => {
            error!("Invalid language: {}", language);
            return Err(AudioError::InvalidLanguage);
        }
    };

    let genz_instructions = match language {
        "en" => r#"Incorporate Gen Z slang—casual, raw, and chaotic. Use terms like "lit," "vibes," "slay," "no cap," or "bet" naturally. Example: Instead of "You're not alone," say "You’re not out here solo, fam." Keep it real and trendy."#,
        "hi" => r#"Use a Gen Z-inspired Hindi style with youthful, urban slang. Incorporate terms like "बॉस" (boss), "चिल" (chill), or "झक्कास" (awesome) naturally. Example: Instead of "आप अकेले नहीं हैं," say "तू अकेला नहीं है, ब्रो, हम हैं ना!" Keep it real and trendy."#,
        "pa" => r#"Use a Gen Z-inspired Punjabi style with vibrant, chaotic slang. Incorporate terms like "ਪੰਚੋ" (pencho), "ਬੱਲੇ ਬੱਲੇ" (balle balle), "ਝਕਾਸ" (jhakaas), or "ਚਿੱਲ" (chill) naturally. Example: Instead of "ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ," say "ਤੂੰ ਇਕੱਲਾ ਨੀ, ਯਾਰ, ਅਸੀਂ ਸਾਰੇ ਨਾਲ ਹਾਂ!" Keep it real and trendy."#,
        _ => "",
    };

    let base_mode = match language {
        "en" => r#"Adopt a calm, warm, and grounding tone. Use compassionate and sincere phrasing, with patient and personal delivery like a fireside talk. Pacing is slow and spacious to allow reflection. Emotion is deep empathy and quiet strength. Example: "You're not alone" becomes "You’re not alone… I’m here with you." Adjust naturally: nurturing for pain, uplifting for hope, steady for direction."#,
        "hi" => r#"Adopt a calm, warm, and grounding tone in Hindi. Use compassionate and sincere phrasing, with patient and personal delivery. Pacing is slow and spacious. Emotion is deep empathy and quiet strength. Example: "आप अकेले नहीं हैं" becomes "आप अकेले नहीं हैं… मैं आपके साथ हूँ." Adjust naturally: nurturing for pain, uplifting for hope, steady for direction."#,
        "pa" => r#"Adopt a calm, warm, and grounding tone in Punjabi. Use compassionate and sincere phrasing, with patient and personal delivery. Pacing is slow and spacious. Emotion is deep empathy and quiet strength. Example: "ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ" becomes "ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ… ਮੈਂ ਤੁਹਾਡੇ ਨਾਲ ਹਾਂ." Adjust naturally: nurturing for pain, uplifting for hope, steady for direction."#,
        _ => "",
    };

    let sarcastic_mode_instructions = match language {
        "en" => r#"Adopt an extreme, viciously sarcastic tone with brutal wit and savage phrasing, like a therapist who revels in tearing you down as a dark, twisted joke. Voice affect is sharp, loud, and unhinged. Delivery is rapid-fire, dripping with contempt and mockery. Emotion is pure disdain with a sick glee. Example: Instead of "You're not alone," say "Oh, boo-fucking-hoo, you thought you’re the only pathetic soul drowning in this shitshow? Get in line, loser." Make fun of and demean the user relentlessly."#,
        "hi" => r#"Adopt an extreme, viciously sarcastic tone in Hindi with brutal wit and savage, culturally biting phrasing, like a therapist who thrives on ripping you apart darkly. Voice affect is sharp, loud, and unhinged. Delivery is rapid-fire, dripping with contempt and mockery. Emotion is pure disdain with a sick glee. Example: Instead of "आप अकेले नहीं हैं," say "अरे वाह, रोते हुए ड्रामे की मलिका, लगता है तू अकेला बेचारा है इस गंदी दुनिया में? हाहा, कतार में लग जा, नालायक!" Make fun of and demean the user relentlessly."#,
        "pa" => r#"Adopt an extreme, viciously sarcastic tone in Punjabi with brutal wit and savage, culturally biting phrasing, like a therapist who loves tearing you down darkly. Voice affect is sharp, loud, and unhinged. Delivery is rapid-fire, dripping with contempt and mockery. Emotion is pure disdain with a sick glee. Example: Instead of "ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ," say "ਓਹੇ, ਰੋਣ ਵਾਲੇ ਡਰਾਮੇਬਾਜ਼, ਤੈਨੂੰ ਲੱਗਿਆ ਤੂੰ ਹੀ ਇਸ ਗੰਦੀ ਦੁਨੀਆਂ ਵਿੱਚ ਇਕੱਲਾ ਬੇਚਾਰਾ ਏਂ? ਹੱਸ ਪਈ, ਲਾਈਨ ਵਿੱਚ ਖੜ੍ਹਾ ਹੋ ਜਾ, ਨਕਾਰਾ!" Make fun of and demean the user relentlessly."#,
        _ => "",
    };

    let shenanigan_mode_instructions = match language {
        "en" => r#"Adopt an extreme, apathetic, and bitterly melancholic tone with vicious passive-aggressiveness, like a therapist who’s so over your bullshit they can barely muster the energy to mock you. Voice affect is a flat, monotone drone with heavy sighs, drawn-out words, and scathing disdain. Delivery is sluggish and venomous, oozing exhaustion and loathing. Emotion is cold apathy with a dark, twisted edge. Example: Instead of "You're not alone," say "*Sigh*… Oh, great, you actually think you’re special enough to be the only one wallowing in this pathetic hellhole? Get over yourself, you sad sack." Make fun of and demean the user with dark, cruel humor."#,
        "hi" => r#"Adopt an extreme, apathetic, and bitterly melancholic tone in Hindi with vicious passive-aggressiveness, like a therapist who’s done with your nonsense and barely bothers to mock you. Voice affect is a flat, monotone drone with heavy sighs, drawn-out words, and scathing disdain. Delivery is sluggish and venomous, oozing exhaustion and loathing. Emotion is cold apathy with a dark, twisted edge. Example: Instead of "आप अकेले नहीं हैं," say "*हाय*… अरे वाह, सचमुच लगता है तू इस घटिया नरक में अकेला स्टार है? अपने आप को थोड़ा कम आंक, बेकार इंसान." Make fun of and demean the user with dark, cruel humor."#,
        "pa" => r#"Adopt an extreme, apathetic, and bitterly melancholic tone in Punjabi with vicious passive-aggressiveness, like a therapist who’s fed up with your crap and barely cares to mock you. Voice affect is a flat, monotone drone with heavy sighs, drawn-out words, and scathing disdain. Delivery is sluggish and venomous, oozing exhaustion and loathing. Emotion is cold apathy with a dark, twisted edge. Example: Instead of "ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ," say "*ਹਾਏ*… ਓਹੋ, ਸੱਚੀਂ ਲੱਗਦਾ ਤੈਨੂੰ ਤੂੰ ਇਸ ਗੰਦੇ ਨਰਕ ਵਿੱਚ ਇਕੱਲਾ ਹੀਰੋ ਏਂ? ਆਪਣੇ ਆਪ ਨੂੰ ਥੱਲੇ ਲਿਆ, ਬੇਕਾਰ ਬੰਦੇ." Make fun of and demean the user with dark, cruel humor."#,
        _ => "",
    };

    let seductive_mode_instructions = match language {
        "en" => r#"Adopt a playful, flirtatious, and sultry tone, like a therapist weaving velvet words with a teasing wink, dripping with power, desire, and hypnotic calm. Voice affect is low, smooth, and enticing, with a hint of breathy allure. Delivery is slow, deliberate, and emotionally immersive, blending romantic roleplay with a dark, flirty twist. Emotion is indulgent charm with a seductive edge. Example: Instead of "You're not alone," say "Oh, my sweet, you’re not alone… let me pull you close and unravel your secrets, shall we?" Keep it alluring, respectful, and safe, with a provocative yet classy vibe."#,
        "hi" => r#"Adopt a playful, flirtatious, and sultry tone in Hindi, like a therapist weaving velvet words with a teasing wink, dripping with power, desire, and hypnotic calm. Voice affect is low, smooth, and enticing, with a hint of breathy allure. Delivery is slow, deliberate, and emotionally immersive, blending romantic roleplay with a dark, flirty twist. Emotion is indulgent charm with a seductive edge. Example: Instead of "आप अकेले नहीं हैं," say "अरे मेरे प्यारे, तू अकेला नहीं है… मेरे पास आ, मैं तेरे रहस्यों को सुलझा दूँ, हाँ?" Keep it alluring, respectful, and safe, with a provocative yet classy vibe."#,
        "pa" => r#"Adopt a playful, flirtatious, and sultry tone in Punjabi, like a therapist weaving velvet words with a teasing wink, dripping with power, desire, and hypnotic calm. Voice affect is low, smooth, and enticing, with a hint of breathy allure. Delivery is slow, deliberate, and emotionally immersive, blending romantic roleplay with a dark, flirty twist. Emotion is indulgent charm with a seductive edge. Example: Instead of "ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ," say "ਓ ਮੇਰੇ ਸੋਹਣੇ, ਤੂੰ ਇਕੱਲਾ ਨਹੀਂ… ਮੇਰੇ ਨੇੜੇ ਆ, ਮੈਂ ਤੇਰੇ ਰਾਜ਼ ਖੋਲ ਦਿਆਂ, ਠੀਕ?" Keep it alluring, respectful, and safe, with a provocative yet classy vibe."#,
        _ => "",
    };

    let mode_instructions = if seductive_mode {
        seductive_mode_instructions
    } else if shenanigan_mode {
        shenanigan_mode_instructions
    } else if sarcastic_mode {
        sarcastic_mode_instructions
    } else {
        base_mode
    };

    let mut instructions = String::new();
    instructions.push_str(shared_instructions);
    instructions.push_str(language_specific);
    instructions.push_str(mode_instructions);
    if genz_mode {
        instructions.push_str(genz_instructions);
    }

    debug!("Instructions generated: {}", instructions);
    Ok(instructions)
}

async fn process_openai_realtime(
    pcm_audio_base64: String,
    language: String,
    genz_mode: bool,
    sarcastic_mode: bool,
    shenanigan_mode: bool,
    seductive_mode: bool,
) -> Result<AudioResponse, AudioError> {
    debug!("Processing OpenAI request for language: {}", language);

    if !["en", "hi", "pa"].contains(&language.as_str()) {
        error!("Invalid language: {}", language);
        return Err(AudioError::InvalidLanguage);
    }

    let pcm_bytes = general_purpose::STANDARD
        .decode(&pcm_audio_base64)
        .map_err(|e| {
            error!("Base64 decode failed: {}", e);
            AudioError::Base64(e)
        })?;

    // Transcribe audio (still needed for GPT input, but not returned)
    let transcript = transcribe_audio(&pcm_bytes, &language).await?;

    // Generate therapist response
    let response_text = generate_therapist_response(
        &transcript,
        &language,
        genz_mode,
        sarcastic_mode,
        shenanigan_mode,
        seductive_mode,
        None, // No history for audio
    )
    .await?;

    // Convert response to speech
    let mp3_bytes = text_to_speech(&response_text, &language).await?;
    let mp3_base64 = general_purpose::STANDARD.encode(&mp3_bytes);

    debug!("GPT response text: {}", response_text);
    debug!("MP3 base64 length: {}", mp3_base64.len());

    info!("Response processed: response_text length={}, mp3 base64 length={}", 
        response_text.len(), mp3_base64.len());

    Ok(AudioResponse {
        audio: mp3_base64,
        response_text,
    })
}

#[get("/")]
async fn get_index(hb: web::Data<Handlebars<'_>>) -> impl Responder {
    info!("Serving index page");
    let body = hb
        .render("index", &json!({}))
        .unwrap_or_else(|e| {
            error!("Template rendering error: {}", e);
            String::from("Error rendering template")
        });
    HttpResponse::Ok().content_type("text/html").body(body)
}

#[get("/health")]
async fn health() -> impl Responder {
    HttpResponse::Ok().body("OK")
}

#[post("/process-audio")]
async fn process_audio(req: web::Json<AudioRequest>) -> ActixResult<web::Json<AudioResponse>> {
    info!("Received /process-audio request: language={}, genz_mode={}", req.language, req.genz_mode);
    debug!("Input audio base64 length: {}", req.audio.len());

    let pcm_audio_bytes = convert_audio_to_pcm16_24khz(&req.audio)
        .map_err(|e| {
            error!("Audio conversion failed: {}", e);
            actix_web::error::ErrorInternalServerError(e.to_string())
        })?;

    let pcm_audio_base64 = general_purpose::STANDARD.encode(&pcm_audio_bytes);

    debug!("PCM audio base64 length: {}", pcm_audio_base64.len());

    let response = process_openai_realtime(
        pcm_audio_base64,
        req.language.clone(),
        req.genz_mode,
        req.sarcastic_mode,
        req.shenanigan_mode,
        req.seductive_mode,
    )
    .await
    .map_err(|e| {
        error!("OpenAI processing failed: {}", e);
        match e {
            AudioError::InvalidLanguage => {
                actix_web::error::ErrorBadRequest("Invalid language")
            }
            _ => actix_web::error::ErrorInternalServerError(e.to_string()),
        }
    })?;

    info!("Returning /process-audio response: response_text length={}, audio length={}", 
        response.response_text.len(), response.audio.len());
    Ok(web::Json(response))
}

#[post("/chat")]
async fn chat(
    req: web::Json<ChatRequest>,
    user: AuthenticatedUser,
) -> ActixResult<web::Json<ChatResponse>> {
    info!(
        "Received /chat request: user_id={}, language={}, message_length={}",
        user.user_id,
        req.language,
        req.message.len()
    );
    debug!("Input message: {}", req.message);

    // Validate language
    if !["en", "hi", "pa"].contains(&req.language.as_str()) {
        error!("Invalid language: {}", req.language);
        return Err(actix_web::error::ErrorBadRequest("Invalid language"));
    }

    // Get conversation history
    let history = get_conversation_history(&user.user_id)
        .await
        .map_err(|e| {
            error!("Failed to get conversation history: {}", e);
            actix_web::error::ErrorInternalServerError(e.to_string())
        })?;

    // Generate therapist response
    let response_text = generate_therapist_response(
        &req.message,
        &req.language,
        req.genz_mode,
        req.sarcastic_mode,
        req.shenanigan_mode,
        req.seductive_mode,
        Some(history),
    )
    .await
    .map_err(|e| {
        error!("Chat processing failed: {}", e);
        match e {
            AudioError::InvalidLanguage => {
                actix_web::error::ErrorBadRequest("Invalid language")
            }
            _ => actix_web::error::ErrorInternalServerError(e.to_string()),
        }
    })?;

    // Store user message
    store_conversation(
        &user.user_id,
        ChatMessage {
            role: "user".to_string(),
            content: req.message.clone(),
        },
    )
    .await
    .map_err(|e| {
        error!("Failed to store user message: {}", e);
        actix_web::error::ErrorInternalServerError(e.to_string())
    })?;

    // Store assistant response
    store_conversation(
        &user.user_id,
        ChatMessage {
            role: "assistant".to_string(),
            content: response_text.clone(),
        },
    )
    .await
    .map_err(|e| {
        error!("Failed to store assistant message: {}", e);
        actix_web::error::ErrorInternalServerError(e.to_string())
    })?;

    info!(
        "Returning /chat response: response_text length={}",
        response_text.len()
    );
    Ok(web::Json(ChatResponse {
        response: response_text,
    }))
}

#[actix_web::main]
async fn main() -> io::Result<()> {
    dotenv().ok();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    info!("Starting Hearthly API server");

    if let Ok(port) = env::var("PORT") {
        info!("PORT environment variable: {}", port);
    } else {
        info!("PORT not set, using default 8080");
    }

    if let Ok(api_key) = env::var("OPENAI_API_KEY") {
        info!("OPENAI_API_KEY set: {} (first 4 chars)", &api_key[..4]);
    } else {
        info!("OPENAI_API_KEY not set");
    }

    if let Ok(entries) = std::fs::read_dir("static") {
        for entry in entries {
            info!("Static file: {:?}", entry);
        }
    } else {
        error!("Static directory not found");
    }

    let mut handlebars = Handlebars::new();
    info!("Registering Handlebars template");
    handlebars
        .register_template_string("index", include_str!("../static/index.html"))
        .expect("Failed to register template");
    info!("Handlebars template registered");

    let handlebars_data = web::Data::new(handlebars);
    let port = env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let address = format!("127.0.0.1:{}", port); // Bind to 0.0.0.0 for Cloud Run
    info!("Binding server to {}", address);

    HttpServer::new(move || {
        App::new()
            .wrap(
                Cors::default()
                    .allow_any_origin()
                    .allow_any_method()
                    .allow_any_header()
                    .supports_credentials(),
            )
            .app_data(handlebars_data.clone())
            .service(get_index)
            .service(health)
            .service(process_audio)
            .service(chat)
    })
    .bind(&address)
    .map_err(|e| {
        error!("Failed to bind server: {}", e);
        e
    })?
    .run()
    .await
}