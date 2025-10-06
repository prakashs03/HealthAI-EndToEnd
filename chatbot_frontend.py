# chatbot_frontend.py
import os
import re
import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import datetime

# voice libs (optional)
try:
    import speech_recognition as sr
    from gtts import gTTS
    from pydub import AudioSegment
    from pydub.playback import play
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False

# Gemini client (optional)
GEMINI_KEY = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None) if st._is_running_with_streamlit else os.environ.get("GEMINI_API_KEY")
gemini_available = False
try:
    import google.generativeai as genai
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        gemini_available = True
except Exception:
    # library not installed / key not set
    gemini_available = False

ROOT = Path(__file__).parent
LOG_PATH = ROOT / "data" / "feedback" / "feedback_log.csv"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def simple_fallback_answer(query):
    """Very small fallback — not ideal. For offline demos only."""
    q = query.lower()
    if "heart" in q or "chest" in q:
        return "Early signs of heart disease include chest discomfort, shortness of breath, fatigue, and swelling. See a doctor for diagnosis."
    if "diabet" in q or "sugar" in q:
        return "Frequent urination, increased thirst, weight loss, and fatigue are common signs of diabetes. Consult a clinician for testing."
    return "Sorry — I don't have a detailed answer offline. Please enable an API key for Gemini in Streamlit secrets for rich answers."

def log_feedback(query, answer, user_feedback=None):
    ts = datetime.utcnow().isoformat()
    row = {"timestamp": ts, "query": query, "answer": answer, "feedback": user_feedback}
    df = pd.DataFrame([row])
    if LOG_PATH.exists():
        df_existing = pd.read_csv(LOG_PATH)
        df_out = pd.concat([df_existing, df], ignore_index=True)
    else:
        df_out = df
    df_out.to_csv(LOG_PATH, index=False)

def call_gemini(query):
    """Call Google Generative API using google.generativeai if available."""
    if not gemini_available:
        return None
    try:
        # simple text generation call
        response = genai.generate_text(model="models/gemini-2.5-pro", prompt=query, max_output_tokens=400)
        return response.text
    except Exception as e:
        st.warning(f"Gemini call failed: {e}")
        return None

def speak_text(text):
    if not VOICE_AVAILABLE:
        st.warning("Voice libs not available (gTTS/pydub). Install them to enable voice output.")
        return
    try:
        tts = gTTS(text=text, lang="en")
        tmp = Path("tmp_tts.mp3")
        tts.save(str(tmp))
        # Use pydub to play if available
        sound = AudioSegment.from_file(str(tmp), format="mp3")
        play(sound)
        tmp.unlink(missing_ok=True)
    except Exception as e:
        st.warning(f"Could not play audio: {e}")

def healthcare_chatbot_component(root: Path):
    st.write("Ask any health-related query in English or Tamil. If you want richer answers enable Gemini API key in Streamlit secrets.")
    col1, col2 = st.columns([4,1])
    query = ""
    with col1:
        query = st.text_input("Ask your health question (type here):", key="chat_query")
    with col2:
        if VOICE_AVAILABLE:
            if st.button("🎙 Speak"):
                # record short audio using speech_recognition (microphone)
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    st.info("Listening... speak now")
                    audio = r.listen(source, timeout=5, phrase_time_limit=8)
                try:
                    txt = r.recognize_google(audio)
                    st.session_state["chat_query"] = txt
                    st.success("Recognized speech: " + txt)
                except Exception as e:
                    st.error(f"Speech recognition error: {e}")

    if st.button("Ask"):
        if not query:
            st.warning("Please type or speak a query.")
            return
        # If Tamil detection needed—Gemini can handle multi-language. We'll forward as-is.
        answer = None
        if gemini_available:
            with st.spinner("Fetching answer from Gemini..."):
                answer = call_gemini(query)
        if not answer:
            answer = simple_fallback_answer(query)
        st.markdown("**Answer:**")
        st.info(answer)
        # speak optional
        if st.checkbox("Play answer (voice)", value=False):
            speak_text(answer)
        # Logging
        fb = st.radio("Was this answer helpful?", ("Select", "Yes", "No"))
        if fb != "Select":
            log_feedback(query, answer, fb)
            st.success("Feedback logged. Thank you!")
