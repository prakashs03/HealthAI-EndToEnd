# translator_app.py
import streamlit as st
import os
from pathlib import Path

try:
    from deep_translator import GoogleTranslator
    DEEP_AVAILABLE = True
except Exception:
    DEEP_AVAILABLE = False

# Gemini availability check — same approach as chatbot file
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
try:
    import google.generativeai as genai
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except Exception:
    GEMINI_AVAILABLE = False

def translator_component(root=None):
    st.write("Translator (English <-> Tamil). Use Gemini if API key set for higher-quality translation.")
    col1, col2 = st.columns([3,1])
    with col1:
        text = st.text_area("Enter text to translate", value="")
        direction = st.radio("Direction", ("EN -> TA", "TA -> EN"))
    if st.button("Translate"):
        if GEMINI_AVAILABLE:
            # simple wrapper prompt - for short translations
            prompt = f"Translate the following text to Tamil:\n\n{text}" if direction=="EN -> TA" else f"Translate the following text to English:\n\n{text}"
            try:
                resp = genai.generate_text(model="models/gemini-2.5-pro", prompt=prompt, max_output_tokens=200)
                st.success(resp.text)
            except Exception as e:
                st.error(f"Gemini error: {e}")
        else:
            if DEEP_AVAILABLE:
                tgt = "tamil" if direction=="EN -> TA" else "english"
                try:
                    translated = GoogleTranslator(source='auto', target=tgt).translate(text)
                    st.success(translated)
                except Exception as e:
                    st.error(f"Translation error: {e}")
            else:
                st.warning("No translation engine available. Install deep_translator or provide GEMINI_API_KEY in secrets.")
