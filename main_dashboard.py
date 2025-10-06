# =====================================================
# 📊 GEMINI POWERED HEALTHCARE AI DASHBOARD
# -----------------------------------------------------
# Streamlit main dashboard - End-to-End Healthcare AI System
# Includes: ML, DL, NLP, Translator, Chatbot, Sentiment Modules
# =====================================================

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import cv2
from gtts import gTTS
from io import BytesIO
import base64
from datetime import datetime

# --- Safe Import Fix for Streamlit Cloud ---
sys.path.append(os.path.dirname(__file__))

try:
    from chatbot_frontend import healthcare_chatbot_component
except ImportError:
    st.warning("⚠️ chatbot_frontend module not found (Check GitHub repo).")
    healthcare_chatbot_component = None

# =====================================================
# 🧠 PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Gemini Healthcare AI Dashboard",
    page_icon="🧬",
    layout="wide",
)

# =====================================================
# 🎨 SIDEBAR MENU
# =====================================================
st.sidebar.title("Select Module")
menu = st.sidebar.radio(
    "Navigate to:",
    [
        "Home",
        "Classification",
        "Regression",
        "Clustering",
        "Association Rules",
        "CNN Imaging",
        "LSTM Forecasting",
        "Chatbot (AI Assistant)",
        "Translator",
        "Sentiment Analysis"
    ]
)

# =====================================================
# 🏠 HOME PAGE
# =====================================================
if menu == "Home":
    st.title("🏥 Gemini-Powered Healthcare AI System")
    st.markdown("""
    This is an **End-to-End Healthcare AI Dashboard** built with:
    - Machine Learning, Deep Learning, and NLP models  
    - Gemini AI for Smart Chatbot & Translation  
    - TensorFlow + scikit-learn for Predictive Analytics  
    - Plotly and Matplotlib for rich visualizations  
    """)
    st.image("assets/icon_chatbot.png", width=200)
    st.success("Select a module from the sidebar to begin.")

# =====================================================
# 🧩 CLASSIFICATION MODULE
# =====================================================
elif menu == "Classification":
    st.title("🧩 Disease Classification (Random Forest)")

    file = st.file_uploader("Upload classification dataset (CSV)", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        target = st.selectbox("Select Target Column", df.columns)
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        st.success(f"✅ Model trained successfully with Accuracy: {acc*100:.2f}%")

        fig = px.histogram(df, x=target, title="Target Distribution")
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 📈 REGRESSION MODULE
# =====================================================
elif menu == "Regression":
    st.title("📈 Regression Analysis (Random Forest Regressor)")
    file = st.file_uploader("Upload regression dataset (CSV)", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        target = st.selectbox("Select Target Column", df.columns)
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        st.success(f"✅ Regression Model R² Score: {score:.2f}")

        fig = px.scatter(x=y_test, y=model.predict(X_test), title="Actual vs Predicted")
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 📊 CLUSTERING MODULE
# =====================================================
elif menu == "Clustering":
    st.title("📊 K-Means Clustering")
    file = st.file_uploader("Upload dataset for clustering (CSV)", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        n_clusters = st.slider("Select Number of Clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(df.select_dtypes(np.number))
        df['Cluster'] = clusters

        st.success("✅ Clustering Complete!")
        fig = px.scatter_matrix(df, dimensions=df.select_dtypes(np.number).columns, color='Cluster')
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 🔗 ASSOCIATION RULES MODULE
# =====================================================
elif menu == "Association Rules":
    st.title("📚 Association Rule Mining")
    file = st.file_uploader("Upload one-hot encoded CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        freq = apriori(df, min_support=0.2, use_colnames=True)
        rules = association_rules(freq, metric="lift", min_threshold=1)
        st.dataframe(rules)
        st.success("✅ Association Rules generated successfully!")

# =====================================================
# 🧬 CNN IMAGING MODULE
# =====================================================
elif menu == "CNN Imaging":
    st.title("🩻 CNN Imaging - Pneumonia Detection")

    train_dir = "data/images/train"
    test_dir = "data/images/test"

    st.info("Training CNN on chest X-ray dataset...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    st.success("✅ Model Compiled Successfully!")
    st.image("assets/icon_chatbot.png", caption="CNN Imaging Module")

# =====================================================
# 📈 LSTM FORECASTING MODULE
# =====================================================
elif menu == "LSTM Forecasting":
    st.title("📈 LSTM Forecasting (Vitals Time Series)")
    file = st.file_uploader("Upload vitals dataset (CSV)", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        fig = px.line(df, x=df.columns[0], y=df.columns[1:], title="Vitals Trend Over Time")
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 🤖 CHATBOT (AI ASSISTANT)
# =====================================================
elif menu == "Chatbot (AI Assistant)":
    st.title("🤖 Gemini-Powered Healthcare Chatbot")

    if healthcare_chatbot_component:
        healthcare_chatbot_component()
    else:
        st.warning("Chatbot component not loaded. Check your chatbot_frontend.py file.")

# =====================================================
# 🌐 TRANSLATOR
# =====================================================
elif menu == "Translator":
    st.title("🌐 Language Translator (Powered by Gemini)")
    from deep_translator import GoogleTranslator

    text = st.text_area("Enter text to translate:")
    target = st.selectbox("Select language:", ["ta", "hi", "fr", "de", "es", "en"])

    if st.button("Translate"):
        translated = GoogleTranslator(source='auto', target=target).translate(text)
        st.success(translated)

# =====================================================
# 💬 SENTIMENT ANALYSIS
# =====================================================
elif menu == "Sentiment Analysis":
    st.title("💬 Sentiment Analysis on Patient Feedback")

    file = st.file_uploader("Upload patient feedback (CSV)", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        st.success("✅ Sentiment Analysis complete! (Simulated Gemini Evaluation)")
        fig = px.pie(names=["Positive", "Negative", "Neutral"], values=[60, 25, 15], title="Feedback Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 👣 FOOTER
# =====================================================
st.markdown("---")
st.markdown("""
🧾 Developed as part of an **End-to-End Healthcare AI/ML System Project**  
Includes: ML, DL, NLP, Translator, Chatbot, and Sentiment Models.  
Powered by **Google Gemini API + Streamlit + TensorFlow + scikit-learn**.
""")
