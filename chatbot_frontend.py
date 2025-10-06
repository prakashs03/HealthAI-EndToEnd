# chatbot_frontend.py
# Chatbot helper: calls Google Gemini if available, otherwise uses a simple fallback.
import streamlit as st
import re
import time

# Try to import the official Google generative ai client
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

def ensure_genai():
    if not GENAI_AVAILABLE:
        return False
    if "GEMINI_API_KEY" not in st.secrets:
        return False
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        return True
    except Exception:
        return False

GENAI_READY = ensure_genai()

def query_gemini(prompt, model="models/gemini-2.5-pro", max_output_tokens=300, temperature=0.2):
    """
    Call Gemini (if available). Returns text or raises exception.
    """
    if not GENAI_READY:
        raise RuntimeError("Gemini not configured/available")

    # The official API may vary; this is the standard client usage.
    try:
        # `generate` returns a response object. The API may differ by version.
        resp = genai.generate(model=model, temperature=temperature, max_output_tokens=max_output_tokens, input=prompt)
        # response.text or resp.candidates[0].output? different versions exist.
        # Try common response fields:
        if hasattr(resp, "text"):
            return resp.text
        if hasattr(resp, "candidates"):
            return resp.candidates[0].output
        # new versions may use resp.output[0].content[0].text
        # Fallback to string repr
        return str(resp)
    except Exception as e:
        raise

def simple_fallback_answer(prompt):
    """
    Very small rule-based fallback if Gemini is not usable.
    Keep it short (one-two lines) — per your UI brief.
    """
    prompt_low = prompt.lower()
    if "heart" in prompt_low or "chest" in prompt_low or "cardio" in prompt_low:
        return "Early signs: chest discomfort, shortness of breath, unexplained fatigue. See a provider for diagnosis."
    if "diabetes" in prompt_low or "blood sugar" in prompt_low:
        return "Common signs: increased thirst, frequent urination, unexplained weight loss. Get a blood test for confirmation."
    if "exercise" in prompt_low or "exercise for heart" in prompt_low:
        return "Moderate aerobic exercise like brisk walking is excellent; aim for 150 mins/week. Check with your doctor first."
    # default
    return "I can help summarize clinical topics briefly. Please ask a specific question (e.g., 'early signs of heart disease')."

def healthcare_chatbot_query(query_text, short_answer=True):
    """
    Central function used by the dashboard.
    short_answer True => produce a short 1-2 line answer.
    If Gemini is available, call it and request short answer. Otherwise fallback.
    """
    # Detect Tamil (approx)
    is_tamil = bool(re.search(r'[\u0B80-\u0BFF]', str(query_text)))

    if GENAI_READY:
        # ask for short or long form
        form = "short" if short_answer else "detailed"
        prompt = f"Provide a {form} clinically-correct, respectfully worded answer to the user question. If the user asked in Tamil, reply in Tamil. Question:\n\n{query_text}\n\nAnswer:"
        try:
            text = query_gemini(prompt, max_output_tokens=300, temperature=0.2)
            # Clean text
            return text.strip()
        except Exception:
            # fallback to simple
            return simple_fallback_answer(query_text)
    else:
        # no Gemini -> simple fallback
        ans = simple_fallback_answer(query_text)
        # If Tamil requested, return in English-only fallback (or very short Tamil phrase)
        if is_tamil:
            # short translation via tiny maps (just a couple of phrases)
            if "heart" in query_text.lower():
                return "முதலுக்கு: மார்புவலி, மூச்சுத்திணறல், சோர்வு. மருத்துவரை அணுகவும்."
            return ans
        return ans
