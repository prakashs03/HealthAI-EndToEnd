# translator_app.py
import streamlit as st

# Try Gemini translation if available
try:
    import google.generativeai as genai
    GEMINI = True
except Exception:
    GEMINI = False

def translate_text(text, target="ta"):
    """Translate text using Gemini if present, default target Tamil (ta)."""
    if GEMINI:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
        if not api_key:
            return "⚠️ Gemini key not found. Please set GEMINI_API_KEY in Streamlit secrets."
        genai.configure(api_key=api_key)
        prompt = f"Translate the following to Tamil: {text}"
        try:
            out = genai.generate_text(model=st.secrets.get("GEMINI_MODEL","models/gemini-2.5-pro"), prompt=prompt)
            return out.text if hasattr(out, "text") else str(out)
        except Exception as e:
            return f"Gemini translation error: {e}\n\nFallback: {fallback_translate(text, target)}"
    else:
        return fallback_translate(text, target)

def fallback_translate(text, target="ta"):
    # Very small rule-based fallback (not real translation).
    if target == "ta":
        return "(Translation unavailable offline) " + text
    return text
