# sentiment_app.py
# Lightweight sentiment demo using a simple lexicon and fallback to Gemini if available.
import streamlit as st
import pandas as pd
from textblob import TextBlob

def sentiment_from_text(text):
    """
    Very small sentiment detector — TextBlob polarity fallback.
    Returns label, score.
    """
    if not text or not str(text).strip():
        return "NEUTRAL", 0.0
    tb = TextBlob(text)
    score = tb.sentiment.polarity  # -1 .. +1
    if score > 0.2:
        return "POSITIVE", float(score)
    if score < -0.2:
        return "NEGATIVE", float(score)
    return "NEUTRAL", float(score)

def sentiment_module_ui():
    st.header("Sentiment Analysis (Patient Feedback)")
    st.write("Paste a patient feedback or upload `patient_feedback.csv` to run sentiment analysis.")
    uploaded = st.file_uploader("Upload CSV (two cols: id,text) for batch sentiment", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if 'text' not in df.columns:
            st.error("CSV must have a 'text' column.")
            return
        df['sentiment_label'], df['sentiment_score'] = zip(*df['text'].apply(sentiment_from_text))
        st.dataframe(df[['text','sentiment_label','sentiment_score']].head(200))
        # simple counts
        c = df['sentiment_label'].value_counts().to_dict()
        st.write("Counts:", c)
    else:
        txt = st.text_area("Type one feedback to analyze")
        if st.button("Analyze"):
            label, score = sentiment_from_text(txt)
            st.success(f"Label: {label}  —  Score: {score:.3f}")
