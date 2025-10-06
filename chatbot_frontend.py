import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os

# ----------------------------
# Configure Gemini
# ----------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ Gemini API key not found! Please add it in Streamlit secrets.")
else:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ----------------------------
# Chatbot Component
# ----------------------------
def healthcare_chatbot_component():
    st.subheader("🤖 Gemini HealthBot")

    # Text input
    user_query = st.text_input("Ask a health-related question:")

    # Voice input
    if st.button("🎙 Speak"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Listening...")
            audio = recognizer.listen(source)
        try:
            user_query = recognizer.recognize_google(audio)
            st.success(f"You said: {user_query}")
        except:
            st.error("❌ Could not understand your voice input.")

    # Gemini AI response
    if user_query:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(user_query)
            answer = response.text.strip()
            st.write("💡 **Answer:**")
            st.write(answer)

            # Voice output
            tts = gTTS(text=answer, lang="en")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tts.save(tmp.name)
                audio_bytes = open(tmp.name, "rb").read()
                st.audio(audio_bytes, format="audio/mp3")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
