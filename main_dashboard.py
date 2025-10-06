import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from mlxtend.frequent_patterns import apriori, association_rules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Embedding
import os
from PIL import Image
import tempfile
import google.generativeai as genai
from gtts import gTTS
import speech_recognition as sr
from io import BytesIO
import base64

# -----------------------------------------------------------
# 🔐 Load Gemini API Key (from Streamlit Secrets)
# -----------------------------------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ Gemini API key not found! Please add it in Streamlit Secrets.")
    st.stop()
else:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# -----------------------------------------------------------
# 🏠 Page Setup
# -----------------------------------------------------------
st.set_page_config(
    page_title="HealthAI: End-to-End Healthcare Platform",
    layout="wide",
)

# -----------------------------------------------------------
# 🖼️ Safe Image Loader
# -----------------------------------------------------------
def load_icon(filename):
    try:
        icon_path = os.path.join(os.path.dirname(__file__), "assets", filename)
        if os.path.exists(icon_path):
            return Image.open(icon_path)
        else:
            st.warning(f"⚠️ File not found: {filename}")
            return None
    except Exception as e:
        st.warning(f"⚠️ Unable to load {filename}. Error: {e}")
        return None


# Load Icons
chatbot_icon = load_icon("icon_chatbot.png")
sentiment_icon = load_icon("icon_sentiment.png")
translator_icon = load_icon("icon_translator.png")

# -----------------------------------------------------------
# 🧠 Sidebar Navigation
# -----------------------------------------------------------
st.sidebar.header("📊 Select Module")
module = st.sidebar.radio(
    "Choose an Analysis Module",
    [
        "Home",
        "Classification",
        "Regression",
        "Clustering",
        "Association Rules",
        "CNN Imaging",
        "Chatbot (AI Assistant)",
        "Translator",
        "Sentiment Analysis",
    ],
)

# -----------------------------------------------------------
# 🏥 Home
# -----------------------------------------------------------
if module == "Home":
    st.title("🏥 HealthAI: End-to-End AI/ML Healthcare Platform")
    if chatbot_icon:
        st.image(chatbot_icon, width=200)

    st.write(
        """
        Welcome to the **HealthAI Dashboard**, your all-in-one intelligent system for medical analytics.  
        Powered by **Gemini AI + Streamlit + TensorFlow + scikit-learn**.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if chatbot_icon:
            st.image(chatbot_icon, width=150)
        st.caption("💬 Chatbot Assistant")
    with col2:
        if sentiment_icon:
            st.image(sentiment_icon, width=150)
        st.caption("🧠 Sentiment Analysis")
    with col3:
        if translator_icon:
            st.image(translator_icon, width=150)
        st.caption("🌐 Translator")

# -----------------------------------------------------------
# 🧩 Classification
# -----------------------------------------------------------
elif module == "Classification":
    st.header("🧩 Disease Classification (Random Forest)")

    uploaded_file = st.file_uploader("📂 Upload CSV dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        target_col = st.selectbox("🎯 Select Target Column", df.columns)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.success(f"✅ Model Accuracy: {acc:.2f}")
        fig = px.bar(x=y_test, y=preds, labels={"x": "Actual", "y": "Predicted"}, title="Actual vs Predicted")
        st.plotly_chart(fig)

# -----------------------------------------------------------
# 📈 Regression
# -----------------------------------------------------------
elif module == "Regression":
    st.header("📈 Patient Health Prediction (Regression)")

    uploaded_file = st.file_uploader("📂 Upload CSV dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        target_col = st.selectbox("🎯 Select Target Column", df.columns)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        st.success(f"✅ Mean Squared Error: {mse:.2f}")
        fig = px.scatter(x=y_test, y=preds, labels={"x": "Actual", "y": "Predicted"}, title="Actual vs Predicted")
        st.plotly_chart(fig)

# -----------------------------------------------------------
# 🧬 Clustering
# -----------------------------------------------------------
elif module == "Clustering":
    st.header("🧬 Patient Data Clustering (K-Means)")

    uploaded_file = st.file_uploader("📂 Upload CSV dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        n_clusters = st.slider("🔢 Select number of clusters", 2, 10, 3)
        model = KMeans(n_clusters=n_clusters, n_init=10)
        df["Cluster"] = model.fit_predict(df.select_dtypes(include=np.number))
        st.dataframe(df.head())
        fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color="Cluster", title="Cluster Visualization")
        st.plotly_chart(fig)

# -----------------------------------------------------------
# 🧾 Association Rules
# -----------------------------------------------------------
elif module == "Association Rules":
    st.header("📚 Association Rule Mining")
    uploaded_file = st.file_uploader("📂 Upload one-hot encoded CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        try:
            freq = apriori(df.astype(bool), min_support=0.2, use_colnames=True)
            rules = association_rules(freq, metric="lift", min_threshold=1)
            st.success("✅ Rules Generated Successfully!")
            st.dataframe(rules.head())
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# -----------------------------------------------------------
# 🩻 CNN Imaging
# -----------------------------------------------------------
elif module == "CNN Imaging":
    st.header("🩻 Medical Imaging (CNN Demo)")
    uploaded_img = st.file_uploader("📷 Upload X-ray Image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        st.image(uploaded_img, caption="Uploaded X-ray", width=300)
        st.success("✅ Image Uploaded Successfully (Mock CNN Pipeline)")

# -----------------------------------------------------------
# 💬 Chatbot (AI Assistant)
# -----------------------------------------------------------
elif module == "Chatbot (AI Assistant)":
    st.header("💬 HealthAI Chatbot")

    text_input = st.text_input("💭 Type your query:")
    voice_btn = st.button("🎤 Speak")

    if text_input:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(text_input)
        st.success(response.text)
        tts = gTTS(response.text)
        tts.save("response.mp3")
        st.audio("response.mp3")

    elif voice_btn:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎙 Listening...")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"🗣 You said: {text}")
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(text)
            st.success(response.text)
            tts = gTTS(response.text)
            tts.save("response.mp3")
            st.audio("response.mp3")
        except:
            st.error("⚠️ Voice input failed. Please try again.")

# -----------------------------------------------------------
# 🌐 Translator
# -----------------------------------------------------------
elif module == "Translator":
    st.header("🌐 Health Document Translator")
    text = st.text_area("Enter text to translate:")
    lang = st.text_input("Target Language (e.g., 'ta' for Tamil, 'hi' for Hindi):")
    if st.button("Translate"):
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Translate this text to {lang}: {text}")
        st.success(response.text)

# -----------------------------------------------------------
# 🧠 Sentiment Analysis
# -----------------------------------------------------------
elif module == "Sentiment Analysis":
    st.header("🧠 Patient Feedback Sentiment Analysis")
    text = st.text_area("🗨 Enter feedback:")
    if st.button("Analyze Sentiment"):
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Analyze sentiment of this text: {text}")
        st.success(response.text)
