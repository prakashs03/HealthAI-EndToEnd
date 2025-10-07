# sentiment_app.py
# Sentiment UI: reads feedback CSV from data/feedback and analyzes sentiment per row.
# Uses Gemini if available, else falls back to TextBlob.

import streamlit as st
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FEEDBACK_CSV = os.path.join(DATA_DIR, "feedback", "patient_feedback.csv")
# try to import gemini wrapper (same as chatbot)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
    try:
        genai.configure(api_key=st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else os.environ.get("GEMINI_API_KEY"))
    except Exception:
        pass
except Exception:
    GENAI_AVAILABLE = False

# fallback to TextBlob
FALLBACK_TEXTBLOB = False
try:
    from textblob import TextBlob
    FALLBACK_TEXTBLOB = True
except Exception:
    FALLBACK_TEXTBLOB = False

def _gemini_sentiment_reason(text: str):
    # ask Gemini to classify sentiment and give a short reason (1-2 lines)
    model = st.secrets.get("GEMINI_MODEL", "models/gemini-2.5-pro") if hasattr(st, "secrets") else os.environ.get("GEMINI_MODEL","models/gemini-2.5-pro")
    prompt = f"Classify the sentiment of this patient feedback as Positive, Neutral or Negative and provide a 1-2 line reason. Feedback: '''{text}'''"
    try:
        resp = genai.generate_text(model=model, prompt=prompt, max_output_tokens=200)
        return resp.text
    except Exception:
        # try alternate responses API
        try:
            resp = genai.responses.generate(model=model, input=prompt)
            # join output
            if hasattr(resp, "output"):
                if isinstance(resp.output, list):
                    return " ".join([o.get("content","") for o in resp.output if isinstance(o, dict)])
            return str(resp)
        except Exception:
            raise

def analyze_sentiment_text(text: str):
    if GENAI_AVAILABLE:
        try:
            out = _gemini_sentiment_reason(text)
            return out
        except Exception:
            pass
    if FALLBACK_TEXTBLOB:
        tb = TextBlob(text)
        polarity = tb.sentiment.polarity
        if polarity > 0.2:
            label = "Positive"
        elif polarity < -0.2:
            label = "Negative"
        else:
            label = "Neutral"
        reason = f"{label} — TextBlob polarity {polarity:.2f}"
        return reason
    # as final fallback
    return "Neutral — cannot analyze (no gemini/textblob installed)."

def sentiment_module_ui():
    st.subheader("Patient feedback sentiment analysis")
    st.write("You can upload `data/feedback/patient_feedback.csv` with a 'text' column or use the bundled file.")
    uploaded = st.file_uploader("Upload feedback CSV (must have 'text' column)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        if os.path.exists(FEEDBACK_CSV):
            df = pd.read_csv(FEEDBACK_CSV)
            st.info("Loaded data/feedback/patient_feedback.csv")
        else:
            st.info("No feedback CSV found. Upload one to analyze, or create sample file at data/feedback/patient_feedback.csv")
            return
    if "text" not in df.columns:
        st.error("CSV must contain a 'text' column.")
        return
    st.dataframe(df.head(10))
    if st.button("Analyze sentiments (this may call Gemini)"):
        with st.spinner("Analyzing..."):
            results = []
            for t in df["text"].astype(str).tolist():
                try:
                    res = analyze_sentiment_text(t)
                except Exception:
                    res = "Error analyzing"
                results.append(res)
            df["sentiment_summary"] = results
            st.dataframe(df.head(20))
            st.download_button("Download results CSV", df.to_csv(index=False), "feedback_with_sentiment.csv")
