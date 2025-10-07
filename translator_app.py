# translator_app.py
# Translator UI - uses Gemini if available for better contextual translations.
import os
import streamlit as st

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
    try:
        genai.configure(api_key=st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else os.environ.get("GEMINI_API_KEY"))
    except Exception:
        pass
except Exception:
    GENAI_AVAILABLE = False

# fallback: googletrans
FALLBACK_GT = False
try:
    from googletrans import Translator as GTTrans
    FALLBACK_GT = True
    _gt = GTTrans()
except Exception:
    FALLBACK_GT = False

def translate_with_gemini(text: str, target_lang: str = "English"):
    model = st.secrets.get("GEMINI_MODEL", "models/gemini-2.5-pro") if hasattr(st, "secrets") else os.environ.get("GEMINI_MODEL","models/gemini-2.5-pro")
    prompt = f"Translate the following text to {target_lang}. Keep medical terms accurate. Text: '''{text}'''"
    try:
        resp = genai.generate_text(model=model, prompt=prompt, max_output_tokens=400)
        return resp.text
    except Exception:
        try:
            resp = genai.responses.generate(model=model, input=prompt)
            if hasattr(resp, "output"):
                if isinstance(resp.output, list):
                    return " ".join([o.get("content","") for o in resp.output if isinstance(o, dict)])
            return str(resp)
        except Exception:
            raise

def translator_ui():
    st.subheader("Translator (Gemini preferred, fallback to googletrans)")
    text = st.text_area("Text to translate", height=120)
    target = st.selectbox("Target language", ["English", "Hindi", "Tamil", "Spanish", "French", "Telugu"])
    if st.button("Translate"):
        if not text.strip():
            st.error("Enter text to translate.")
        else:
            if GENAI_AVAILABLE:
                try:
                    out = translate_with_gemini(text, target)
                    st.success("Gemini translation:")
                    st.write(out)
                except Exception as e:
                    st.warning("Gemini translation failed, falling back.")
                    if FALLBACK_GT:
                        st.write(_gt.translate(text, dest=target.lower()).text)
                    else:
                        st.error("No fallback translator available.")
            else:
                if FALLBACK_GT:
                    st.write(_gt.translate(text, dest=target.lower()).text)
                else:
                    st.error("No translation available — install googletrans or provide GEMINI_API_KEY.")
