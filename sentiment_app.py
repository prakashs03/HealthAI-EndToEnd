# sentiment_app.py
import streamlit as st
from textblob import TextBlob

def sentiment_module_ui():
    st.subheader("🧾 Sentiment Analysis (Patient Feedback)")
    text = st.text_area("Paste patient feedback or reviews here:")
    if st.button("Analyze Sentiment"):
        if not text.strip():
            st.warning("Please enter some feedback.")
            return
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        if polarity > 0.1:
            label = "Positive"
        elif polarity < -0.1:
            label = "Negative"
        else:
            label = "Neutral"
        st.write(f"**Sentiment:** {label} (score: {polarity:.2f})")
