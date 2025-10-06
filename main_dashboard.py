import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, mean_squared_error
from mlxtend.frequent_patterns import apriori, association_rules
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.preprocessing import image
import cv2, os
from PIL import Image
import tempfile
from gtts import gTTS
import base64
import plotly.graph_objects as go
from io import BytesIO

# ✅ Import chatbot module (updated)
from chatbot_frontend import healthcare_chatbot_component

# ✅ Streamlit config
st.set_page_config(
    page_title="HealthAI: End-to-End Healthcare AI System",
    layout="wide",
    page_icon="💊"
)

# ✅ Header
st.title("🏥 HealthAI: End-to-End AI/ML Healthcare Platform")
st.markdown("""
Welcome to the **HealthAI Dashboard**, your all-in-one intelligent system for medical analytics.  
Powered by **Gemini AI + Streamlit + TensorFlow + scikit-learn**.
""")

# Sidebar Navigation
st.sidebar.header("📂 Modules")
module = st.sidebar.radio("Choose a module", [
    "🏠 Home",
    "📈 Regression (Outcome Prediction)",
    "🧬 Classification (Disease Risk)",
    "🔬 Clustering (Patient Segmentation)",
    "📚 Association Rule Mining",
    "🩻 CNN (Imaging Diagnostics)",
    "📊 LSTM (Time Series)",
    "💬 Chatbot (Gemini)",
])

# =========================
# 🏠 HOME
# =========================
if module == "🏠 Home":
    st.image("assets/icon_chatbot.png", width=200)
    st.success("✅ All modules ready. Choose one from the left sidebar.")

# =========================
# 📈 REGRESSION
# =========================
elif module == "📈 Regression (Outcome Prediction)":
    st.header("📈 Predict Length of Stay (Regression Model)")

    uploaded_file = st.file_uploader("Upload regression CSV with `los` column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        numeric_df = df.select_dtypes(include=[np.number])
        if 'los' not in numeric_df.columns:
            st.error("⚠️ Missing 'los' column for prediction!")
        else:
            X = numeric_df.drop('los', axis=1)
            y = numeric_df['los']

            model = LinearRegression()
            model.fit(X, y)
            preds = model.predict(X)

            st.success(f"✅ Model trained successfully! RMSE: {np.sqrt(mean_squared_error(y, preds)):.2f}")
            fig = px.scatter(x=y, y=preds, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted LOS")
            st.plotly_chart(fig, use_container_width=True)

# =========================
# 🧬 CLASSIFICATION
# =========================
elif module == "🧬 Classification (Disease Risk)":
    st.header("🧬 Disease Risk Classification")

    uploaded_file = st.file_uploader("Upload classification CSV (with 'label' column)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        if 'label' not in df.columns:
            st.error("⚠️ 'label' column missing!")
        else:
            X = df.drop('label', axis=1).select_dtypes(include=[np.number])
            y = df['label']

            model = LogisticRegression(max_iter=200)
            model.fit(X, y)
            preds = model.predict(X)
            acc = accuracy_score(y, preds)

            st.success(f"✅ Model Accuracy: {acc*100:.2f}%")
            st.plotly_chart(px.histogram(x=preds, title="Predicted Class Distribution"))

# =========================
# 🔬 CLUSTERING
# =========================
elif module == "🔬 Clustering (Patient Segmentation)":
    st.header("🔬 Patient Segmentation using K-Means")

    uploaded_file = st.file_uploader("Upload clustering CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        numeric_df = df.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric_df)

        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled)

        score = silhouette_score(scaled, df['Cluster'])
        st.success(f"✅ Clustering completed. Silhouette Score: {score:.3f}")

        fig = px.scatter(df, x=numeric_df.columns[0], y=numeric_df.columns[1], color='Cluster',
                         title="Patient Clusters")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# 📚 ASSOCIATION RULE MINING
# =========================
elif module == "📚 Association Rule Mining":
    st.header("📚 Mining Comorbidity Associations (Apriori)")

    uploaded_file = st.file_uploader("Upload one-hot encoded transactions CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        if df.select_dtypes(exclude=['number']).shape[1] > 0:
            st.error("⚠️ Non-numeric data found. Ensure one-hot encoded input.")
        else:
            freq = apriori(df, min_support=0.2, use_colnames=True)
            rules = association_rules(freq, metric="lift", min_threshold=1.0)
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# =========================
# 🩻 CNN (IMAGING)
# =========================
elif module == "🩻 CNN (Imaging Diagnostics)":
    st.header("🩻 CNN-based Pneumonia Detection")

    img = st.file_uploader("Upload a Chest X-Ray Image", type=["jpg", "png"])
    if img:
        temp = Image.open(img).convert("RGB").resize((128, 128))
        img_array = np.expand_dims(np.array(temp)/255.0, axis=0)

        cnn = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        st.image(temp, caption="Uploaded X-Ray", width=300)
        st.success("✅ CNN model ready for predictions (demo mode).")

# =========================
# 📊 LSTM (Time Series)
# =========================
elif module == "📊 LSTM (Time Series)":
    st.header("📊 Patient Vital Trends Prediction (LSTM)")

    uploaded_file = st.file_uploader("Upload time series data (vitals.csv)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.line_chart(df.set_index(df.columns[0]))
        st.info("✅ Data visualized. Use actual vitals for LSTM training in full setup.")

# =========================
# 💬 CHATBOT (Gemini)
# =========================
elif module == "💬 Chatbot (Gemini)":
    st.header("💬 Healthcare Chatbot with Voice & Text (Gemini AI)")
    healthcare_chatbot_component()
