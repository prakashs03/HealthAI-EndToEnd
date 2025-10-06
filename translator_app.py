# translator_app.py
import streamlit as st
from transformers import MarianTokenizer, MarianMTModel
import torch

def translator_ui():
    st.subheader("🌐 Translator (EN ↔ other languages)")
    text = st.text_area("Enter text to translate (English input):")
    tgt = st.selectbox("Translate to:", ["ta", "hi", "es", "fr", "de"])
    if st.button("Translate"):
        if not text.strip():
            st.warning("Please enter text.")
            return
        model_name = f"Helsinki-NLP/opus-mt-en-{tgt}"
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            batch = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
            translated = model.generate(**batch)
            out = tokenizer.decode(translated[0], skip_special_tokens=True)
            st.success(out)
        except Exception as e:
            st.error("Translation model not available in this environment. Ensure internet access and that transformers is installed.")
