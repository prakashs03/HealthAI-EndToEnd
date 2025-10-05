import streamlit as st
from deep_translator import GoogleTranslator
import re

# ===============================================================
# PAGE CONFIG
# ===============================================================
st.set_page_config(
    page_title="🌐 Healthcare Translator (English ↔ தமிழ்)",
    page_icon="🌍",
    layout="centered"
)

st.markdown(
    """
    <h2 style='text-align:center;color:#007BFF;'>🌐 Healthcare Translator (English ↔ தமிழ்)</h2>
    <p style='text-align:center;'>Bridge the language gap between doctors and patients</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ===============================================================
# HELPER FUNCTION
# ===============================================================
def detect_language(text):
    """Detects if text is Tamil or English."""
    if bool(re.search(r'[\u0B80-\u0BFF]', text)):
        return "ta"
    return "en"


def translate_text(text):
    """Translates between Tamil and English automatically."""
    if not text.strip():
        return "⚠️ Please enter text to translate."

    lang = detect_language(text)

    try:
        if lang == "en":
            translated = GoogleTranslator(source="en", target="ta").translate(text)
            direction = "English → Tamil"
        else:
            translated = GoogleTranslator(source="ta", target="en").translate(text)
            direction = "Tamil → English"
        return f"**{direction}:**\n\n{translated}"

    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# ===============================================================
# UI
# ===============================================================
st.subheader("📝 Enter text below to translate:")
user_text = st.text_area("Input text:", height=150, placeholder="Type here in English or Tamil...")

if st.button("🔄 Translate"):
    with st.spinner("Translating... please wait..."):
        output = translate_text(user_text)
        st.markdown(output)

st.markdown(
    """
    <hr>
    <div style='text-align:center;color:gray;'>
        <b>Powered by Deep Translator | Healthcare AI Project</b>
    </div>
    """,
    unsafe_allow_html=True
)
