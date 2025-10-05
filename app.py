# ================================================================
#  HEALTHCARE CHATBOT API  (Tamil + English)
# ================================================================
# Requirements:
# pip install fastapi uvicorn google-generativeai deep-translator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from deep_translator import GoogleTranslator
import google.generativeai as genai
import re

# ================================================================
#  API CONFIGURATION
# ================================================================
app = FastAPI(
    title="Gemini Healthcare Chatbot API",
    description="A multilingual healthcare chatbot using with Tamil + English support",
    version="2.0"
)

# Allow Streamlit or browser access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
#  GEMINI MODEL CONFIG
# ================================================================
# Replace YOUR_API_KEY with your Gemini API key
genai.configure(api_key="AIzaSyCs-iW346inNJ0Pmc-PidcM2L4NOH9C7o4")

MODEL_NAME = "models/gemini-2.5-pro"

# ================================================================
#  CHATBOT FUNCTION
# ================================================================
def healthcare_chatbot(query_text: str):
    """Process Tamil or English question and return Gemini answer."""
    try:
        # Detect Tamil text
        is_tamil = bool(re.search(r'[\u0B80-\u0BFF]', query_text))

        # Translate to English if Tamil
        if is_tamil:
            query_en = GoogleTranslator(source="ta", target="en").translate(query_text)
        else:
            query_en = query_text

        # Create Gemini model
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            f"Provide a concise, factual medical answer for this health-related question: {query_en}. "
            f"Do not include unnecessary conversation or examples."
        )

        # Extract Gemini answer
        answer_en = response.text.strip()

        # If Tamil input, translate back to Tamil
        if is_tamil:
            answer_ta = GoogleTranslator(source="en", target="ta").translate(answer_en)
            return {"language": "tamil", "answer": answer_ta}
        else:
            return {"language": "english", "answer": answer_en}

    except Exception as e:
        return {"error": str(e)}

# ================================================================
#  FASTAPI ENDPOINTS
# ================================================================
@app.get("/")
def root():
    return {"message": "✅ Gemini Healthcare Chatbot API is running!"}

@app.get("/ask")
def ask_health_query(q: str = Query(..., description="Health-related question in Tamil or English")):
    """Main endpoint to handle chatbot queries"""
    result = healthcare_chatbot(q)
    return result

# ================================================================
#  RUN COMMAND
# ================================================================
# Use:  python -m uvicorn app:app --reload --port 8000
# ================================================================
