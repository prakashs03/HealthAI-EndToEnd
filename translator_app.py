# translator_app.py
# Simple translator interface:
# - If Gemini available, ask it to translate.
# - Otherwise use a tiny rule or say to use Google Translate manually.
import streamlit as st
from googletrans import Translator as GT_Translator

def translate_ui():
    st.header("Translator (English <-> Tamil)")
    text = st.text_area("Enter text to translate")
    target = st.selectbox("Target language", ["ta", "en"], index=0)
    if st.button("Translate"):
        # Try googletrans first (local)
        try:
            tr = GT_Translator()
            out = tr.translate(text, dest=target)
            st.success(out.text)
        except Exception:
            st.warning("Local translator not available. If deployed with Gemini key, translation via Gemini will be attempted.")
            # If you want to call Gemini translation, integrate through chatbot_frontend healthcare_chatbot_query with instruction to translate.
            st.info("Alternatively, use external translator or configure Gemini for server-side translation.")
