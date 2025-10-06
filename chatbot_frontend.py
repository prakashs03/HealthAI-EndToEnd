# chatbot_frontend.py
import os
from typing import Optional

GENAI_READY = False

# We do NOT run any Streamlit commands at import-time to avoid page-config errors.

def _try_call_gemini(question: str, api_key: str, model: str="models/gemini-2.5-pro") -> Optional[str]:
    """
    Try to invoke Google Generative API if package present and key provided.
    Returns text or raises an exception.
    """
    try:
        import google.generativeai as genai  # optional package
    except Exception as e:
        raise RuntimeError("google.generativeai not installed.") from e

    if not api_key:
        raise RuntimeError("No API key provided for Gemini.")

    genai.configure(api_key=api_key)
    # Note: depending on your installed genai library, method names differ.
    # We attempt a simple call; adapt this call to your library version if necessary.
    try:
        # newer libs may provide genai.generate_text or genai.generate
        if hasattr(genai, "generate_text"):
            resp = genai.generate_text(model=model, prompt=question, max_output_tokens=512)
            return getattr(resp, "text", str(resp))
        elif hasattr(genai, "generate"):
            resp = genai.generate(model=model, input=question)
            # The return shape can differ; try to extract text
            if hasattr(resp, "candidates"):
                return resp.candidates[0].content
            return str(resp)
        else:
            raise RuntimeError("Unsupported genai client version.")
    except Exception as e:
        raise RuntimeError("Gemini call failed: " + str(e)) from e

def healthcare_chatbot_query(question: str) -> str:
    """
    Public function called by Streamlit main app.
    Attempts Gemini if API key set in environment variable GEMINI_API_KEY,
    otherwise returns a friendly fallback answer.
    """
    key = os.environ.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        # No API key -> fallback canned answers
        q = question.lower()
        if "heart" in q:
            return ("**Brief:** Chest pain/pressure, shortness of breath, fatigue, dizziness can be early signs of heart disease. "
                    "If severe or sudden, seek emergency care. Ask 'explain more' for details.")
        if "diabetes" in q:
            return ("**Brief:** Frequent urination, increased thirst, unexplained weight loss, tiredness. "
                    "Consult a doctor for testing.")
        return ("**Brief:** I can provide general health info. For personal medical advice, see a clinician. "
                "Ask 'explain more' to get a longer description.")
    # If key present, try gemini
    try:
        text = _try_call_gemini(question, key)
        return text
    except Exception as e:
        return f"❗ Gemini unavailable: {e}\n\nFallback (brief): For medical concerns, consult a health professional."
