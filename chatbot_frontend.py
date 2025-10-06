# chatbot_frontend.py
import streamlit as st
import re

# try to import google.generativeai (Gemini) - optional
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

def _genai_response(question: str) -> str:
    """Return response using Gemini if available and key present; otherwise fallback."""
    # If secrets has GEMINI_API_KEY and library is present, call it.
    try:
        if GENAI_AVAILABLE and "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = "models/gemini-2.5-flash"  # a supported model on the user's list
            resp = genai.generate_text(model=model, text=question)
            # new genai clients may differ; attempt multiple attributes
            text = ""
            if hasattr(resp, "text"):
                text = resp.text
            elif isinstance(resp, dict) and "candidates" in resp:
                text = resp["candidates"][0].get("content", "")
            else:
                text = str(resp)
            return text.strip()
    except Exception as e:
        # fallback below
        pass

    # Fallback simple answer: take keywords and answer concisely
    q = question.lower()
    if "heart" in q or "chest" in q or "heart disease" in q:
        return "Early signs: chest discomfort, breathlessness, unusual fatigue — consult a doctor if present."
    if "diabetes" in q:
        return "Common signs: frequent urination, thirst, fatigue, unexplained weight loss; see provider for tests."
    if re.search(r'[\u0B80-\u0BFF]', question):  # Tamil unicode detection
        # short Tamil fallback
        return "முன் அறிகுறிகள்: மார்பு வலி, மூச்சுத் திணறல் மற்றும் சோர்வு — மருத்துவரை சந்திக்கவும்."
    return "I don't have internet access here; please ask a simple medical question or enable Gemini API."

def healthcare_chatbot_component():
    st.subheader("💬 Healthcare Chatbot")
    st.write("Type a question below (the model will reply concisely). If you want more detail, add `Please explain more.`")
    user_q = st.text_input("Ask a health question (text):")
    if st.button("Get Answer"):
        if not user_q.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Getting answer..."):
                answer = _genai_response(user_q)
            # default behavior: concise one- or two-line answer. If user appended 'explain', return longer.
            if "explain" in user_q.lower() or "detail" in user_q.lower():
                st.info(answer)
            else:
                # Show concise: first sentence only
                concise = answer.split(".")[0]
                if concise.strip() == "":
                    concise = answer
                st.success(concise.strip())
