# chatbot_frontend.py
# Chatbot wrapper that tries to use Google Gemini via google-generativeai.
# Falls back to a lightweight rule-based responder when Gemini isn't available.

import os
import re
import streamlit as st

GEMINI_KEY = None
try:
    # read GEMINI_API_KEY from Streamlit secrets or environment
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else os.environ.get("GEMINI_API_KEY")
except Exception:
    GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

# try import official google generative library
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
except Exception:
    GENAI_AVAILABLE = False

# Choose default model name (you can change via secrets)
DEFAULT_MODEL = st.secrets.get("GEMINI_MODEL", "models/gemini-2.5-pro") if hasattr(st, "secrets") else os.environ.get("GEMINI_MODEL","models/gemini-2.5-pro")

def _is_explain_request(prompt: str) -> bool:
    return bool(re.search(r"\b(explain|how|why|describe|detail|details|tell me more)\b", prompt, flags=re.I))

def _shortify(text: str, max_lines: int = 2) -> str:
    # simple truncation for short answers
    lines = text.strip().splitlines()
    if len(lines) <= max_lines:
        # also restrict to ~280 chars
        out = " ".join(lines)
        return out if len(out) < 450 else out[:450].rsplit(" ",1)[0] + "..."
    else:
        return " ".join(lines[:max_lines])[:450] + "..."

def _gemini_query(prompt: str, explain: bool=False) -> str:
    if not GENAI_AVAILABLE:
        raise RuntimeError("google-generativeai package not installed.")
    # Use simple generate_text API
    gen_model = DEFAULT_MODEL
    try:
        response = genai.generate_text(model=gen_model, prompt=prompt, max_output_tokens=400)
        out = response.text
        return out
    except Exception as e:
        # try alternate call for newer lib versions
        try:
            response = genai.responses.generate(model=gen_model, input=prompt)
            if hasattr(response, "output") and response.output:
                # some response objects have output_text or output[0].content
                if isinstance(response.output, list):
                    return " ".join([c.get("content", "") for c in response.output if isinstance(c, dict)])
                else:
                    return getattr(response, "output_text", str(response))
        except Exception as e2:
            raise e

def healthcare_chatbot_query(prompt: str) -> str:
    """
    Returns a 1-2 line answer by default. If user asks to explain (keywords),
    returns more detailed text. Uses Gemini if available; otherwise returns fallback.
    """
    is_explain = _is_explain_request(prompt)
    # if gemini available and API key set -> call
    if GENAI_AVAILABLE and GEMINI_KEY:
        try:
            # build a short system instruction to get concise answers
            instruction = ("You are a concise healthcare assistant. Provide safe, factual answers. "
                           "If the user asks for 'explain' or 'how' give a longer answer. Cite nothing.")
            final_prompt = instruction + "\n\nUser: " + prompt
            out = _gemini_query(final_prompt, explain=is_explain)
            if not is_explain:
                return _shortify(out, max_lines=2)
            else:
                return out
        except Exception as e:
            # fallback to simple logic
            fallback = _fallback_response(prompt, is_explain)
            return fallback + " (fallback: Gemini call failed)"
    else:
        # fallback
        fallback = _fallback_response(prompt, is_explain)
        return fallback + " (fallback: offline mode)"

def _fallback_response(prompt: str, explain: bool):
    # very small domain-limited fallback mapping for key topics (safe, non-diagnostic)
    p = prompt.lower()
    if "heart" in p or "chest pain" in p:
        base = "Common early signs of heart disease include chest discomfort, shortness of breath, unusual fatigue, and palpitations."
        if explain:
            base += " These symptoms vary by individual; women may experience jaw/back discomfort, nausea, or extreme fatigue. Seek immediate care for severe chest pain or shortness of breath."
        return _shortify(base, max_lines=4 if explain else 2)
    if "sugar" in p or "diabetes" in p:
        base = "To reduce sugar intake: read labels, avoid sugary drinks, choose whole fruits, and prefer unsweetened products."
        if explain:
            base += " Replace sugary drinks with water, plan meals, and consult a dietitian for personalized recommendations."
        return _shortify(base, max_lines=4 if explain else 2)
    # default
    default = "I'm a HealthAI assistant. For brief answers ask normally. For more detail include 'explain'."
    if explain:
        default += " I can summarize clinical guidelines and lifestyle advice but cannot replace a clinician."
    return default
