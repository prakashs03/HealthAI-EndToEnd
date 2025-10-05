import streamlit as st
import google.generativeai as genai
from deep_translator import GoogleTranslator
import re

# ---------------------------
# 🌟 CONFIGURATION
# ---------------------------
API_KEY = "AIzaSyCs-iW346inNJ0Pmc-PidcM2L4NOH9C7o4"  # your Gemini API key
genai.configure(api_key=API_KEY)

# ---------------------------
# ⚙️ Initialize Model
# ---------------------------
model = genai.GenerativeModel("models/gemini-2.5-pro")

# ---------------------------
# 🧠 Sentiment Function
# ---------------------------
def analyze_sentiment(text):
    # Detect Tamil text
    is_tamil = bool(re.search(r'[\u0B80-\u0BFF]', str(text)))
    
    # Translate Tamil to English for analysis
    if is_tamil:
        translated_text = GoogleTranslator(source='ta', target='en').translate(text)
    else:
        translated_text = text

    # Ask Gemini for sentiment
    prompt = f"Classify the sentiment of this text as Positive, Negative, or Neutral: '{translated_text}'. Give a one-line reason."
    
    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
    except Exception as e:
        result = f"⚠️ Error: {str(e)}"
        return result

    # Translate back if input was Tamil
    if is_tamil:
        translated_back = GoogleTranslator(source='en', target='ta').translate(result)
        return translated_back
    else:
        return result

# ---------------------------
# 🎨 Streamlit UI
# ---------------------------
st.set_page_config(page_title="Healthcare Sentiment Analyzer", page_icon="❤️", layout="centered")

st.title("🧠 Healthcare Sentiment Analyzer")
st.write("Analyze patient feedback or clinical text sentiment in **English or Tamil**.")

st.markdown("---")

# Input text box
text_input = st.text_area("📝 Enter feedback or statement here:", height=150)

if st.button("🔍 Analyze Sentiment"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing sentiment... please wait ⏳"):
            sentiment = analyze_sentiment(text_input)

        # Display output
        st.markdown("### 🧾 Result:")
        st.success(sentiment)

        # Visual indicator
        if any(word.lower() in sentiment.lower() for word in ["positive", "நல்ல", "சிறந்த", "மகிழ்ச்சி"]):
            st.markdown("🟢 **Sentiment: Positive** 😊")
        elif any(word.lower() in sentiment.lower() for word in ["negative", "மோசமான", "தவறான", "துக்கம்"]):
            st.markdown("🔴 **Sentiment: Negative** 😔")
        else:
            st.markdown("🟡 **Sentiment: Neutral** 😐")

st.markdown("---")
st.caption("© 2025 Healthcare AI Project | Built with ❤️ using Google Gemini + Streamlit")
