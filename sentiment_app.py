# sentiment_app.py
import os
from pathlib import Path
import streamlit as st
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
FEEDBACK_LOG = DATA_DIR / "feedback" / "feedback_log.csv"

def sentiment_component(root=None):
    st.write("Sentiment Analysis of feedback. Uses logged chat feedback and patient_feedback.csv.")
    p = FEEDBACK_LOG
    if p.exists():
        df = pd.read_csv(p)
        st.write("Recent feedback (tail):")
        st.dataframe(df.tail(20))
        # quick counts
        if "feedback" in df.columns:
            counts = df["feedback"].value_counts(dropna=True)
            st.bar_chart(counts)
    else:
        st.info("No feedback log found yet (feedback_log.csv). Use the Chatbot and submit feedback to populate.")

    # allow manual upload of patient feedback CSV for batch sentiment
    uploaded = st.file_uploader("Upload patient_feedback.csv (optional)", type=["csv"])
    if uploaded is not None:
        df2 = pd.read_csv(uploaded)
        st.write(df2.head())
        st.success("File loaded. For labeling/sentiment you can download and analyze in notebook; this demo does a simple polarity via TextBlob if installed.")
        try:
            from textblob import TextBlob
            def pol(x):
                return TextBlob(str(x)).sentiment.polarity
            df2["polarity"] = df2.iloc[:,0].apply(pol)
            st.write(df2.head())
            st.bar_chart(df2["polarity"].apply(lambda v: "pos" if v>0 else ("neg" if v<0 else "neu")).value_counts())
        except Exception as e:
            st.warning("TextBlob not installed on server. Install `textblob` to run quick sentiment demo.")
