# chatbot_frontend.py
# Wrapper to call Gemini (if available) or fallback to a small local responder.
import os
import streamlit as st
import json
GENIE_AVAILABLE = False

# Try to import Google Generative AI client if available (user must install and configure)
try:
    import google.generativeai as genai
    GENIE_AVAILABLE = True
except Exception:
    GENIE_AVAILABLE = False

def _gemini_query(prompt, model="models/gemini-2.5-pro"):
    """Call the Gemini API via google.generativeai if available."""
    if not GENIE_AVAILABLE:
        raise RuntimeError("Gemini SDK not installed")
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        raise RuntimeError("Gemini API key missing in Streamlit secrets")
    genai.configure(api_key=api_key)
    response = genai.generate_text(model=model, prompt=prompt, max_output_tokens=512)
    if hasattr(response, "text"):
        return response.text
    # fallback
    return str(response)

def healthcare_chatbot_query(query, short=True, explain=False):
    """
    Query Gemini if available. If short=True, request 1-2 line answer.
    If explain=True, request detailed answer.
    If Gemini not available, run fallback simple responses.
    """
    sys_prompt = "You are a medical assistant. Provide accurate, concise health information. Include disclaimers."
    instruction = sys_prompt + "\nUser: " + query
    if short:
        instruction += "\nAnswer in 1-2 lines."
    if explain:
        instruction += "\nThen provide a detailed explanation with bullet points."

    if GENIE_AVAILABLE:
        try:
            text = _gemini_query(instruction, model=st.secrets.get("GEMINI_MODEL", "models/gemini-2.5-pro"))
            return text
        except Exception as e:
            return f"⚠️ Gemini error: {e}\n\nFallback answer:\n" + _local_bot(query, short, explain)
    else:
        return _local_bot(query, short, explain)

def _local_bot(query, short=True, explain=False):
    """Lightweight fallback: keyword-based answers (works offline)."""
    q = query.lower()
    if "heart" in q and "symptom" in q or "sign" in q:
        ans = "Chest discomfort, shortness of breath, and unexplained fatigue are common early signs of heart disease."
    elif "sugar" in q or "diabetes" in q or "prevent" in q:
        ans = "Reduce added sugars, monitor portions, stay active, and consult a dietitian."
    elif "covid" in q or "pneumonia" in q:
        ans = "Fever, cough, and difficulty breathing — seek medical attention if severe."
    else:
        ans = "Sorry, I don't have a full answer offline. Please ask again or enable Gemini for comprehensive answers."
    if explain:
        ans += "\n\nExplanation:\n- Keep healthy weight\n- Exercise regularly\n- Monitor key vitals and consult clinician for tests."
    return ans
