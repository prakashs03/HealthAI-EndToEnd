# translator_app.py
def translator_ui():
    import streamlit as st
    st.write("Translator (English <-> Tamil). Uses `deep_translator` if installed.")
    text = st.text_area("Enter text to translate")
    direction = st.radio("Translate to", ("English", "Tamil"))
    if st.button("Translate"):
        if not text.strip():
            st.warning("Enter text.")
            return
        try:
            from deep_translator import GoogleTranslator
            if direction == "English":
                out = GoogleTranslator(source='auto', target='en').translate(text)
            else:
                out = GoogleTranslator(source='auto', target='ta').translate(text)
            st.success("Translation:")
            st.write(out)
        except Exception as e:
            st.error("Translation failed (install deep_translator). Fallback below.")
            st.write(text)
