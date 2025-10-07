# app.py
# FastAPI backend for HealthAI project
# Provides REST endpoints for chatbot, translator, and sentiment modules
# Uses Gemini 2.5 Pro or Gemini 2.5 Flash (fallbacks included)
# Run locally: uvicorn app:app --reload

import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Try importing Gemini
GENAI_AVAILABLE = False
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
try:
    import google.generativeai as genai
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# Fallback: textblob / googletrans if needed
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception:
    TEXTBLOB_AVAILABLE = False

try:
    from googletrans import Translator
    translator = Translator()
    GT_AVAILABLE = True
except Exception:
    GT_AVAILABLE = False

# FastAPI app instance
app = FastAPI(
    title="HealthAI Backend API",
    description="REST API for Chatbot, Translator, and Sentiment modules using Gemini 2.5 Pro / Flash",
    version="1.0.0"
)

# ------------- Utility functions -------------

def _shortify(text: str, max_chars=300):
    text = re.sub(r'\s+', ' ', text.strip())
    return text if len(text) <= max_chars else text[:max_chars].rsplit(' ', 1)[0] + "..."

def _is_explain_request(prompt: str):
    return bool(re.search(r"\b(explain|why|how|describe|details?)\b", prompt, flags=re.I))

# Gemini core generator
def _gemini_generate(prompt: str, model="models/gemini-2.5-pro", max_tokens=400):
    if not GENAI_AVAILABLE:
        raise RuntimeError("Gemini not available")
    try:
        response = genai.generate_text(model=model, prompt=prompt, max_output_tokens=max_tokens)
        return response.text
    except Exception as e:
        raise RuntimeError(str(e))

# ------------- Chatbot Endpoint -------------

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chatbot_endpoint(req: ChatRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query text.")
    explain = _is_explain_request(query)
    try:
        if GENAI_AVAILABLE and GEMINI_KEY:
            system_prompt = (
                "You are HealthAI — a concise healthcare assistant. "
                "Provide safe, factual medical information. "
                "Give short 1–2 line answers unless the user asks to explain."
            )
            text = _gemini_generate(f"{system_prompt}\nUser: {query}", model="models/gemini-2.5-pro")
            return {"response": text if explain else _shortify(text)}
        else:
            # fallback simple rules
            p = query.lower()
            if "heart" in p:
                return {"response": "Heart disease symptoms include chest pain and fatigue. Seek medical advice."}
            if "sugar" in p or "diabetes" in p:
                return {"response": "Control sugar with diet, exercise, and doctor supervision."}
            return {"response": "HealthAI assistant offline fallback: basic health tips only."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {e}")

# ------------- Translator Endpoint -------------

class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "English"

@app.post("/translate")
def translate_endpoint(req: TranslateRequest):
    text = req.text.strip()
    target = req.target_lang.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text to translate.")
    try:
        if GENAI_AVAILABLE and GEMINI_KEY:
            prompt = f"Translate this text to {target}. Ensure medical terms are accurate: '''{text}'''"
            trans = _gemini_generate(prompt, model="models/gemini-2.5-flash")
            return {"translated_text": trans}
        elif GT_AVAILABLE:
            out = translator.translate(text, dest=target.lower()).text
            return {"translated_text": out}
        else:
            return {"translated_text": "Translation unavailable (no Gemini/googletrans)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {e}")

# ------------- Sentiment Endpoint -------------

class SentimentRequest(BaseModel):
    text: str

@app.post("/sentiment")
def sentiment_endpoint(req: SentimentRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")
    try:
        if GENAI_AVAILABLE and GEMINI_KEY:
            prompt = f"Classify the sentiment of this text (Positive, Neutral, Negative) and explain briefly: '''{text}'''"
            out = _gemini_generate(prompt, model="models/gemini-2.5-pro")
            return {"sentiment_analysis": out}
        elif TEXTBLOB_AVAILABLE:
            tb = TextBlob(text)
            pol = tb.sentiment.polarity
            label = "Positive" if pol > 0.2 else "Negative" if pol < -0.2 else "Neutral"
            return {"sentiment_analysis": f"{label} (polarity={pol:.2f})"}
        else:
            return {"sentiment_analysis": "Neutral — fallback mode."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment error: {e}")

# ------------- Root Info -------------

@app.get("/")
def root():
    return {
        "message": "Welcome to HealthAI Backend API",
        "endpoints": ["/chat", "/translate", "/sentiment"],
        "gemini_enabled": GENAI_AVAILABLE,
        "textblob_enabled": TEXTBLOB_AVAILABLE,
        "googletrans_enabled": GT_AVAILABLE
    }
