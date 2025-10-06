# sentiment_app.py
def sentiment_ui():
    import streamlit as st
    import pandas as pd
    st.write("Sentiment analysis (TextBlob). Upload CSV with one column or paste text.")
    uploaded = st.file_uploader("Upload CSV (one column of text)", type=["csv"], key="sent")
    text_area = st.text_area("Or paste text to analyze")
    texts = []
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            first_col = df.columns[0]
            texts = df[first_col].astype(str).tolist()
            st.dataframe(df.head(10))
        except Exception as e:
            st.error("Failed to read CSV: " + str(e))
            return
    elif text_area.strip():
        texts = [text_area.strip()]
    if st.button("Analyze"):
        if not texts:
            st.warning("Provide text input or CSV.")
            return
        try:
            from textblob import TextBlob
            results = []
            for t in texts:
                tb = TextBlob(t)
                results.append({"text": t, "polarity": tb.sentiment.polarity, "subjectivity": tb.sentiment.subjectivity})
            st.dataframe(pd.DataFrame(results))
        except Exception as e:
            st.error("TextBlob not available. Install via requirements.")
            st.write("Fallback: Could not compute sentiment.")
