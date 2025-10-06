import streamlit as st
from deep_translator import GoogleTranslator
import google.generativeai as genai

# ----------------------------------------------------------
# ✅ STEP 1: Configure Gemini API Key
# ----------------------------------------------------------
try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-pro")
except Exception as e:
        st.warning("⚠️ Gemini API key missing or invalid. Please add GEMINI_API_KEY in Streamlit secrets.")
        gemini_model = None


# ----------------------------------------------------------
# ✅ STEP 2: Translator Function
# ----------------------------------------------------------
def translator_module_ui():
        st.markdown("## 🌐 Medical Translator")
        st.markdown("Bridge the language gap between doctors and patients using AI-powered multilingual translation.")

    # Select translation direction
    col1, col2 = st.columns(2)
    with col1:
                source_lang = st.selectbox("Translate From", ["english", "tamil", "hindi", "telugu", "malayalam", "kannada"])
            with col2:
                        target_lang = st.selectbox("Translate To", ["tamil", "english", "hindi", "telugu", "malayalam", "kannada"])

    # User input
    text_input = st.text_area("🗣 Enter the medical sentence or instruction:", 
                                                            placeholder="Example: Please take this medicine twice a day after food")

    # Translation process
    if st.button("🔁 Translate"):
                if not text_input.strip():
                                st.warning("⚠️ Please enter text before translating.")
                                return

        try:
                        translation = GoogleTranslator(source=source_lang, target=target_lang).translate(text_input)
                        st.success(f"**Translated Text ({target_lang.title()}):**\n\n{translation}")

            # Add Gemini insight — explanation of translation context
            if gemini_model:
                                with st.spinner("💭 Understanding medical context..."):
                                                        prompt = (
                                                                                    f"You are a multilingual medical assistant. "
                                                                                    f"Explain this translated text briefly and ensure it makes medical sense:\n"
                                                                                    f"Original ({source_lang}): {text_input}\n"
                                                                                    f"Translated ({target_lang}): {translation}"
                                                        )
                                                        result = gemini_model.generate_content(prompt)
                                                        st.info(result.text)

except Exception as e:
            st.error(f"❌ Translation failed: {str(e)}")

    st.markdown("---")
    st.caption("💡 Powered by Deep Translator + Gemini AI for context-aware multilingual medical communication.")


# ----------------------------------------------------------
# ✅ STEP 3: Allow Standalone Execution
# ----------------------------------------------------------
if __name__ == "__main__":
        st.set_page_config(page_title="HealthAI Translator", page_icon="🌐")
        translator_module_ui()
