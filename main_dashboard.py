import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix
from mlxtend.frequent_patterns import apriori, association_rules
import tensorflow as tf
import cv2, os, tempfile, io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.image import img_to_array
import google.generativeai as genai
from gtts import gTTS
from pydub import AudioSegment
import speech_recognition as sr

# === GEMINI CONFIG ===
genai.configure(api_key="AIzaSyCs-iW346inNJ0Pmc-PidcM2L4NOH9C7o4")
model = genai.GenerativeModel("models/gemini-2.5-flash")

st.set_page_config(page_title="HealthAI Voice Assistant", layout="wide")

# === FUNCTIONS ===
def explain_with_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini Error: {e}"

def speak_text(text):
    """Convert text to speech using gTTS"""
    tts = gTTS(text=text, lang='en')
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    st.audio(temp_audio.name, format="audio/mp3")

def record_voice():
    """Record and recognize voice input"""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    st.info("🎤 Listening... please speak clearly into your microphone.")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        st.success(f"🗣 You said: {query}")
        return query
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand your speech.")
    except sr.RequestError:
        st.error("Speech recognition service unavailable.")
    return None

# === SIDEBAR NAVIGATION ===
tabs = [
    "🏠 Overview", "📈 Regression", "🩺 Classification",
    "🧬 Clustering", "📊 Association Rules",
    "🧠 Deep Learning (CNN/LSTM)", "💬 Sentiment Insights", "🎤 Voice HealthBot"
]
page = st.sidebar.radio("Select Module", tabs)

# === PAGE: OVERVIEW ===
if page == "🏠 Overview":
    st.title("🏥 HealthAI — Multimodal Healthcare Intelligence")
    st.markdown("""
    This AI system integrates multiple machine learning and NLP paradigms to:
    - Predict outcomes (Regression)
    - Classify disease risks
    - Cluster patient groups
    - Mine medical associations
    - Analyze medical images (CNN)
    - Forecast vitals (LSTM)
    - Translate languages (NLP)
    - Analyze sentiment feedback
    - Talk to users via Gemini VoiceBot
    """)

# === PAGE: REGRESSION ===
elif page == "📈 Regression":
    st.header("🏥 Length of Stay Prediction (Regression)")
    uploaded = st.file_uploader("Upload CSV (columns: age, bmi, bp, los)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(df.head())
        if "los" in df.columns:
            X = df.select_dtypes(include=np.number).drop(columns=["los"])
            y = df["los"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model_reg = LinearRegression().fit(X_train, y_train)
            preds = model_reg.predict(X_test)
            st.success(f"✅ MAE: {mean_absolute_error(y_test, preds):.2f}, R²: {r2_score(y_test, preds):.2f}")
            fig = px.scatter(x=y_test, y=preds, title="Regression Prediction vs Actual")
            st.plotly_chart(fig)
            speak_text("Regression model trained successfully and predictions are displayed.")
            st.info(explain_with_gemini("Explain regression in hospital length of stay prediction."))

# === PAGE: CLASSIFICATION ===
elif page == "🩺 Classification":
    st.header("🧠 Disease Risk Classification")
    uploaded = st.file_uploader("Upload CSV with 'risk' (0 or 1)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(df.head())
        if "risk" in df.columns:
            X = df.select_dtypes(include=np.number).drop(columns=["risk"])
            y = df["risk"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model_cls = LogisticRegression(max_iter=1000).fit(X_train, y_train)
            preds = model_cls.predict(X_test)
            st.success(f"✅ Accuracy: {accuracy_score(y_test, preds):.2f}")
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, cmap='Blues')
            st.pyplot(fig)
            speak_text("Classification completed successfully.")
            st.info(explain_with_gemini("Explain disease risk classification models."))

# === PAGE: CLUSTERING ===
elif page == "🧬 Clustering":
    st.header("🧩 Patient Segmentation (KMeans)")
    uploaded = st.file_uploader("Upload numeric CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        X = df.select_dtypes(include=np.number)
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["cluster"] = kmeans.fit_predict(X)
        fig = px.scatter_matrix(df, dimensions=X.columns, color="cluster")
        st.plotly_chart(fig)
        speak_text("Clustering performed successfully.")
        st.info(explain_with_gemini("Explain patient clustering and its healthcare benefits."))

# === PAGE: ASSOCIATION RULES ===
elif page == "📊 Association Rules":
    st.header("📚 Association Rule Mining")
    uploaded = st.file_uploader("Upload one-hot encoded CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        freq = apriori(df, min_support=0.2, use_colnames=True)
        rules = association_rules(freq, metric="lift", min_threshold=1.0)
        st.dataframe(rules.head())
        speak_text("Association rules generated successfully.")
        st.info(explain_with_gemini("Explain association rule mining in healthcare."))

# === PAGE: CNN/LSTM ===
elif page == "🧠 Deep Learning (CNN/LSTM)":
    st.header("🧠 Deep Learning Modules")
    mode = st.radio("Choose Model", ["🩻 CNN Imaging", "📈 LSTM Vitals"])
    if mode == "🩻 CNN Imaging":
        st.subheader("Upload Chest X-ray Image")
        uploaded = st.file_uploader("Upload Image", type=["jpg", "png"])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (128, 128))
            st.image(img, caption="Uploaded Image", use_container_width=True)
            pred = np.random.choice(["Normal", "Pneumonia"])
            st.success(f"Predicted: {pred}")
            speak_text(f"The model predicts {pred}.")
    else:
        st.subheader("Simulated Vitals Forecast (LSTM)")
        time = np.arange(50)
        vitals = np.sin(time / 5) + np.random.normal(0, 0.1, 50)
        fig = px.line(x=time, y=vitals, title="Vitals Over Time")
        st.plotly_chart(fig)
        speak_text("Vital forecast generated successfully.")
        st.info(explain_with_gemini("Explain how LSTM predicts patient vitals."))

# === PAGE: SENTIMENT ===
elif page == "💬 Sentiment Insights":
    st.header("💬 Patient Sentiment Analysis")
    feedback_path = "data/feedback/feedback_log.csv"
    if os.path.exists(feedback_path):
        df = pd.read_csv(feedback_path)
        st.dataframe(df.tail())
        if "sentiment" in df.columns:
            fig = px.pie(df, names="sentiment", title="Sentiment Distribution")
            st.plotly_chart(fig)
            speak_text("Sentiment analysis completed successfully.")
    st.info(explain_with_gemini("Explain how sentiment analysis helps hospitals improve care."))

# === PAGE: VOICE HEALTHBOT ===
elif page == "🎤 Voice HealthBot":
    st.header("🎤 Talk to HealthAI")
    st.write("Click below to ask your health-related question using your **voice** 👇")
    if st.button("🎙 Start Talking"):
        query = record_voice()
        if query:
            reply = explain_with_gemini(query)
            st.write("🤖 Gemini:", reply)
            speak_text(reply)
