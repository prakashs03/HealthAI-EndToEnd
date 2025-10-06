import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

# ============================================================
# ✅ Load Gemini API key from Streamlit Secrets
# ============================================================
try:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_KEY:
    st.error("❌ Gemini API key not found! Please add it in Streamlit Secrets.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ============================================================
# 🎙️ Function for Speech Input
# ============================================================
def listen_to_user():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now.")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            st.success(f"🗣️ You said: {text}")
            return text
        except sr.UnknownValueError:
            st.warning("Sorry, I couldn’t understand you. Try again!")
            return ""
        except sr.RequestError:
            st.error("Speech recognition service unavailable.")
            return ""

# ============================================================
# 🔊 Function for Voice Output
# ============================================================
def speak_text(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            audio_data = open(fp.name, "rb").read()
        st.audio(audio_data, format="audio/mp3")
    except Exception as e:
        st.warning(f"Audio playback failed: {e}")

# ============================================================
# 💬 Healthcare Chatbot Component
# ============================================================
def healthcare_chatbot_component():
    st.header("🩺 Gemini-Powered Healthcare Chatbot 🤖")
    st.markdown("Ask your **health-related query** in English or Tamil. You can also use your voice!")

    user_query = st.text_input("💬 Type your question below:")
    voice_input = st.button("🎤 Speak instead")

    if voice_input:
        user_query = listen_to_user()

    if user_query:
        with st.spinner("🤖 Thinking..."):
            try:
                response = model.generate_content(user_query)
                st.success("✅ Response:")
                st.markdown(response.text)
                speak_text(response.text)
            except Exception as e:
                st.error(f"Gemini Error: {e}")
