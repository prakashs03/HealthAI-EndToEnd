# chatbot_frontend.py
import os
import streamlit as st

GENAI_READY = False
_gemini_client = None
_model_name = "models/gemini-2.5-pro"  # prefer a supported model

# Try to import google generative AI (if installed in Streamlit environment)
try:
    import google.generativeai as genai
    GEMINI_KEY = None
    # Streamlit secrets first (on Streamlit Cloud)
    try:
        GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_KEY:
        st.warning("❌ Gemini API key not found in secrets or environment. Chatbot will fallback to canned replies.")
    else:
        genai.configure(api_key=GEMINI_KEY)
        _gemini_client = genai
        GENAI_READY = True
except Exception:
    # If import fails, we simply continue — the main app will show helpful message
    GENAI_READY = False

def healthcare_chatbot_query(question: str) -> str:
    """
    Query Gemini if available, otherwise return a fallback helpful response.
    """
    if GENAI_READY and _gemini_client is not None:
        try:
            # Use a simple text generation call (API may differ — adjust to your installed genai client)
            resp = _gemini_client.generate_text(model=_model_name, prompt=question, max_output_tokens=512)
            # .text or similar depending on library version
            answer = getattr(resp, "text", None) or str(resp)
            return answer
        except Exception as e:
            return f"❗ Gemini call failed: {e}\n\nFallback: Here's a brief general answer: " \
                   "For health concerns, consult a clinician. Common signs depend on the condition."
    else:
        # fallback: short helpful template answer (keeps responses safe)
        if "heart" in question.lower():
            return ("**Brief:** Chest pain/pressure, shortness of breath, fatigue, dizziness can be early signs of heart disease. "
                    "If severe or sudden, seek emergency care.\n\n**If you want more detail**, ask 'explain more'.")
        elif "diabetes" in question.lower():
            return ("**Brief:** Frequent urination, increased thirst, hunger, unexplained weight loss, and fatigue. "
                    "Speak to your doctor for testing.")
        else:
            return ("**Brief:** I can provide general health guidance and resources. For specific medical advice, "
                    "please consult a healthcare provider. Ask for more detail if needed.")

# End of chatbot_frontend.py
