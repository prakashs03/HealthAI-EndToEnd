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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image
import numpy as np

# ✅ Import chatbot component
from chatbot_frontend import healthcare_chatbot_component

# ----------------------------
# Streamlit Configuration
# ----------------------------
st.set_page_config(
    page_title="HealthAI Dashboard",
    layout="wide",
    page_icon="💊"
)

st.title("🏥 HealthAI: End-to-End AI/ML Healthcare Platform")
st.markdown("""
Welcome to the **HealthAI Dashboard**, your all-in-one intelligent system for medical analytics.  
Powered by **Gemini AI + Streamlit + TensorFlow + scikit-learn**.
""")

# Sidebar
st.sidebar.header("📂 Modules")
module = st.sidebar.radio(
    "Choose a module",
    [
        "🏠 Home",
        "📈 Regression (Outcome Prediction)",
        "🧬 Classification (Disease Risk)",
        "🔬 Clustering (Patient Segmentation)",
        "📚 Association Rule Mining",
        "🩻 CNN (Imaging Diagnostics)",
        "📊 LSTM (Time Series)",
        "💬 Chatbot (Gemini)",
    ],
)

# ----------------------------
# Modules
# ----------------------------

if module == "🏠 Home":
    st.image("assets/icon_chatbot.png", width=200)
    st.success("✅ All modules are ready. Choose one from the sidebar.")

# =========================
# 📈 REGRESSION
# =========================
elif module == "📈 Regression (Outcome Prediction)":
    st.header("📈 Predict Length of Stay (Regression Model)")

    file = st.file_uploader("Upload regression CSV with `los` column", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write(df.head())

        if "los" not in df.columns:
            st.error("⚠️ Missing 'los' column!")
        else:
            X = df.select_dtypes(include=[np.number]).drop(columns=["los"])
            y = df["los"]

            model = LinearRegression()
            model.fit(X, y)
            preds = model.predict(X)

            rmse = np.sqrt(mean_squared_error(y, preds))
            st.success(f"✅ Model trained. RMSE: {rmse:.2f}")

            fig = px.scatter(x=y, y=preds, labels={"x": "Actual", "y": "Predicted"})
            st.plotly_chart(fig, use_container_width=True)

# =========================
# 🧬 CLASSIFICATION
# =========================
elif module == "🧬 Classification (Disease Risk)":
    st.header("🧬 Disease Risk Classification")

    file = st.file_uploader("Upload classification CSV (with 'label' column)", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write(df.head())

        if "label" not in df.columns:
            st.error("⚠️ 'label' column missing!")
        else:
            X = df.select_dtypes(include=[np.number]).drop(columns=["label"])
            y = df["label"]

            model = LogisticRegression(max_iter=200)
            model.fit(X, y)
            preds = model.predict(X)
            acc = accuracy_score(y, preds)

            st.success(f"✅ Accuracy: {acc*100:.2f}%")
            st.plotly_chart(px.histogram(x=preds, title="Predicted Class Distribution"))

# =========================
# 🔬 CLUSTERING
# =========================
elif module == "🔬 Clustering (Patient Segmentation)":
    st.header("🔬 Patient Segmentation using K-Means")

    file = st.file_uploader("Upload clustering CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write(df.head())

        X = df.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=42)
        df["Cluster"] = kmeans.fit_predict(scaled)

        score = silhouette_score(scaled, df["Cluster"])
        st.success(f"✅ Clustering completed. Silhouette Score: {score:.3f}")

        fig = px.scatter(df, x=X.columns[0], y=X.columns[1], color="Cluster")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# 📚 ASSOCIATION RULES
# =========================
elif module == "📚 Association Rule Mining":
    st.header("📚 Apriori Association Rules")

    file = st.file_uploader("Upload one-hot encoded transactions CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write(df.head())

        try:
            freq = apriori(df, min_support=0.2, use_colnames=True)
            rules = association_rules(freq, metric="lift", min_threshold=1.0)
            st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
        except Exception as e:
            st.error(f"❌ Error: {e}")

# =========================
# 🩻 CNN (IMAGING)
# =========================
elif module == "🩻 CNN (Imaging Diagnostics)":
    st.header("🩻 Pneumonia Detection (CNN Model)")

    file = st.file_uploader("Upload a Chest X-Ray Image", type=["jpg", "png"])
    if file:
        img = Image.open(file).convert("RGB").resize((128, 128))
        st.image(img, caption="Uploaded X-Ray", width=300)
        st.success("✅ Demo CNN ready (mock prediction).")

# =========================
# 💬 CHATBOT (GEMINI)
# =========================
elif module == "💬 Chatbot (Gemini)":
    st.header("💬 Healthcare Chatbot with Text & Voice (Gemini AI)")
    healthcare_chatbot_component()
