# =====================================================
# 🤖 GEMINI HEALTHCARE CHATBOT FRONTEND
# =====================================================

import os
import streamlit as st
import pandas as pd
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import base64
from datetime import datetime
import google.generativeai as genai

# =====================================================
# ⚙️ Gemini API Configuration
# =====================================================
# Make sure you have added this in Streamlit Secrets
# (in Streamlit Cloud: Settings → Secrets → add)
# GEMINI_API_KEY = "your_api_key_here"

if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ Gemini API key not found in Streamlit Secrets. Please add GEMINI_API_KEY.")
else:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# =====================================================
# 🔊 Voice Input Function
# =====================================================
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙 Speak now...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"✅ You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("❌ Could not understand your speech. Please try again.")
            return None
        except sr.RequestError:
            st.error("⚠️ Speech Recognition service error.")
            return None

# =====================================================
# 🔉 Text-to-Speech Function
# =====================================================
def speak_response(text):
    tts = gTTS(text)
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    audio_bytes = audio_fp.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    st.audio(f"data:audio/mp3;base64,{audio_base64}", format="audio/mp3")

# =====================================================
# 🧠 Gemini Chatbot Function
# =====================================================
def gemini_response(user_input):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(user_input)
        return response.text
    except Exception as e:
        st.error(f"⚠️ Gemini API Error: {e}")
        return "I'm having trouble responding right now. Please try again later."

# =====================================================
# 🧾 Feedback Logger
# =====================================================
def log_feedback(user_input, response):
    feedback_dir = "data/feedback"
    os.makedirs(feedback_dir, exist_ok=True)
    log_path = os.path.join(feedback_dir, "feedback_log.csv")

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": user_input,
        "bot_response": response
    }

    if not os.path.exists(log_path):
        pd.DataFrame([entry]).to_csv(log_path, index=False)
    else:
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, pd.DataFrame([entry])], ignore_index=True)
        log_df.to_csv(log_path, index=False)

# =====================================================
# 💬 Chatbot Streamlit UI
# =====================================================
def healthcare_chatbot_component():
    st.header("🩺 Gemini Healthcare Assistant")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_area("💭 Type your health-related question here:", height=100)
    with col2:
        if st.button("🎙 Speak"):
            voice_text = recognize_speech()
            if voice_text:
                user_input = voice_text

    if st.button("Send"):
        if not user_input.strip():
            st.warning("⚠️ Please enter or speak something.")
        else:
            with st.spinner("Gemini is thinking..."):
                response = gemini_response(user_input)
                st.markdown("### 💬 Gemini Response:")
                st.success(response)
                speak_response(response)
                log_feedback(user_input, response)
                st.info("✅ Chat logged successfully!")

    st.markdown("---")
    st.caption("This chatbot is powered by Google Gemini for healthcare assistance.")
