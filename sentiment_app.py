# sentiment_app.py
import os, datetime
import streamlit as st

try:
    from textblob import TextBlob
    TB_OK = True
except Exception:
    TB_OK = False

LOG_PATH = os.path.join("data", "feedback", "feedback_log.csv")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def analyze_sentiment(text):
    # Try TextBlob if installed
    if TB_OK:
        try:
            tb = TextBlob(text)
            polarity = tb.sentiment.polarity
            if polarity > 0.2:
                return "Positive"
            elif polarity < -0.2:
                return "Negative"
            else:
                return "Neutral"
        except Exception:
            pass
    # fallback: very simple keyword-based
    t = text.lower()
    if any(w in t for w in ["good","great","excellent","satisfied","happy"]):
        return "Positive"
    if any(w in t for w in ["bad","terrible","angry","unsatisfied","poor"]):
        return "Negative"
    return "Neutral"

def log_feedback(query, answer, feedback_text):
    # Append to feedback_log.csv
    import csv
    header = ["timestamp","query","answer","feedback"]
    row = [datetime.datetime.utcnow().isoformat(), query, answer, feedback_text]
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
    with open(LOG_PATH, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
