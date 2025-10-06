import streamlit as st
import google.generativeai as genai
import os
import csv
import datetime
from gtts import gTTS
from io import BytesIO
import speech_recognition as sr
from pydub import AudioSegment

# ----------------------------------------------------------
# ✅ STEP 1: Configure Gemini API
# ----------------------------------------------------------
try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-pro")
except Exception as e:
        gemini_model = None
        st.warning("⚠️ Gemini API key missing! Add GEMINI_API_KEY in Streamlit secrets.")


# ----------------------------------------------------------
# ✅ STEP 2: Chatbot Function
# ----------------------------------------------------------
def healthcare_chatbot_component():
        st.markdown("## 🤖 Healthcare Chatbot")
        st.markdown("Ask any health-related question. You can either type or use your voice 🎙️")

    if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

    # Text input
    user_input = st.text_area("💬 Type your question:", placeholder="Example: What are the early symptoms of diabetes?")

    # Voice input
    if st.button("🎙️ Speak your question"):
                recognizer = sr.Recognizer()
                try:
                                with sr.Microphone() as source:
                                                    st.info("Listening... Please speak clearly.")
                                                    audio_data = recognizer.listen(source, timeout=8, phrase_time_limit=10)
                                                    user_input = recognizer.recognize_google(audio_data)
                                                    st.success(f"🎧 You said: {user_input}")
                except Exception as e:
                                st.error("❌ Voice input failed. Please check your microphone permissions.")

            # If user asks a question
            if st.button("🩺 Get Health Advice"):
                        if not user_input.strip():
                                        st.warning("⚠️ Please type or speak a question.")
                                        return

                        # Send to Gemini
                        if gemini_model:
                                        with st.spinner("🤖 Thinking..."):
                                                            prompt = (
                                                                                    f"You are a multilingual medical assistant chatbot. "
                                                                                    f"Provide a short, clear, and medically accurate response. "
                                                                                    f"If the user asks for more details, expand the explanation.\n"
                                                                                    f"User question: {user_input}"
                                                            )
                                                            try:
                                                                                    response = gemini_model.generate_content(prompt)
                                                                                    bot_reply = response.text.strip()

                                                                st.markdown("### 🧠 Gemini's Response:")
                                                                st.write(bot_reply)

                                                # Convert to speech
                                                                tts = gTTS(bot_reply)
                                                                audio_bytes = BytesIO()
                                                                tts.write_to_fp(audio_bytes)
                                                                st.audio(audio_bytes.getvalue(), format="audio/mp3")

                                                # Save chat log
                                                                save_chat_to_csv(user_input, bot_reply)

                                                st.session_state.chat_history.append(("You", user_input))
                                                st.session_state.chat_history.append(("Bot", bot_reply))

except Exception as e:
                    st.error(f"⚠️ Error generating response: {str(e)}")
else:
            st.error("❌ Gemini API not initialized.")


# ----------------------------------------------------------
# ✅ STEP 3: Chat Logger
# ----------------------------------------------------------
def save_chat_to_csv(user_text, bot_reply):
        os.makedirs("feedback", exist_ok=True)
        log_path = "feedback/feedback_log.csv"
        with open(log_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.datetime.now(), user_text, bot_reply])


# ----------------------------------------------------------
# ✅ STEP 4: Run as Standalone
# ----------------------------------------------------------
if __name__ == "__main__":
        st.set_page_config(page_title="HealthAI Chatbot", page_icon="🤖")
        healthcare_chatbot_component()
