# translator_app.py
import streamlit as st

def translator_ui():
    st.markdown("Translator (English <-> Tamil) — automatic translation using Deep Translator or fallback.")
    text = st.text_area("Enter text to translate", "")
    direction = st.radio("Direction", ("Auto -> English", "Auto -> Tamil"))
    if st.button("Translate"):
        if not text.strip():
            st.warning("Enter some text to translate.")
            return
        try:
            from deep_translator import GoogleTranslator
            if direction == "Auto -> English":
                out = GoogleTranslator(source='auto', target='en').translate(text)
            else:
                out = GoogleTranslator(source='auto', target='ta').translate(text)
            st.success("Translated text:")
            st.write(out)
        except Exception as e:
            st.error("Translation failed (missing dependency or network). Fallback message below.")
            st.write("**Fallback:** " + text)
