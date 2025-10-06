# sentiment_app.py
import streamlit as st
import os
import time
import pandas as pd

def sentiment_ui():
    st.markdown("Sentiment Analysis on patient feedback (TextBlob quick demo).")
    uploaded = st.file_uploader("Upload feedback CSV (one column 'text')", type=["csv"], key="sentiment")
    text_input = st.text_area("Or paste feedback text here")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(100))
        texts = df.iloc[:,0].astype(str).tolist()
    else:
        texts = [text_input] if text_input.strip() else []
    if st.button("Analyze Sentiment"):
        if not texts:
            st.warning("Provide input text or upload CSV.")
            return
        try:
            from textblob import TextBlob
            results = []
            for t in texts:
                tb = TextBlob(t)
                pol = tb.sentiment.polarity
                subj = tb.sentiment.subjectivity
                results.append({"text": t, "polarity": pol, "subjectivity": subj})
            rdf = pd.DataFrame(results)
            st.dataframe(rdf)
            st.success("Sentiment computed.")
        except Exception as e:
            st.error("TextBlob not available. Install via requirements or fallback.")
            st.write("Fallback: unable to compute sentiment here.")
