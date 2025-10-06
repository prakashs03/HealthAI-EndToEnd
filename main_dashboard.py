import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os
import cv2
import tempfile
import tensorflow as tf
from mlxtend.frequent_patterns import apriori, association_rules
from chatbot_frontend import healthcare_chatbot_component

# ============================================================
# ✅ Streamlit Page Config (must be first)
# ============================================================
st.set_page_config(
    page_title="HealthAI End-to-End Dashboard",
    layout="wide",
    page_icon="🧠",
)

st.sidebar.title("📊 Select Module")
app_mode = st.sidebar.radio(
    "Choose an Analysis Module",
    ["Home", "Classification", "Regression", "Clustering",
     "Association Rules", "CNN Imaging", "Chatbot (AI Assistant)",
     "Translator", "Sentiment Analysis"]
)

# ============================================================
# 🏠 HOME SECTION
# ============================================================
if app_mode == "Home":
    st.title("🏥 HealthAI: End-to-End AI/ML Healthcare Platform")
    st.markdown("""
    Welcome to the **HealthAI Dashboard**, your all-in-one intelligent system for medical analytics.  
    Powered by **Gemini AI + Streamlit + TensorFlow + scikit-learn**.
    """)
    st.image("assets/icon_chatbot.png", width=200)
    st.success("Select a module from the sidebar to begin!")

# ============================================================
# 📈 REGRESSION (LOS Prediction)
# ============================================================
elif app_mode == "Regression":
    st.header("📈 Regression Analysis (LOS Prediction)")
    uploaded = st.file_uploader("Upload Regression CSV (with numeric columns + target)", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            st.error("Need at least one feature and one target numeric column.")
        else:
            X = numeric_df.iloc[:, :-1]
            y = numeric_df.iloc[:, -1]
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            fig = px.scatter(x=y, y=y_pred, title="Actual vs Predicted", labels={"x": "Actual", "y": "Predicted"})
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Model R² Score: {model.score(X, y):.3f}")

# ============================================================
# 🔍 CLUSTERING (K-Means)
# ============================================================
elif app_mode == "Clustering":
    st.header("🔍 K-Means Clustering")
    uploaded = st.file_uploader("Upload CSV for Clustering", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        numeric_df = df.select_dtypes(include=[np.number])
        n_clusters = st.slider("Select number of clusters", 2, 6, 3)
        km = KMeans(n_clusters=n_clusters, random_state=42)
        df["Cluster"] = km.fit_predict(numeric_df)
        fig = px.scatter_3d(df, x=numeric_df.columns[0], y=numeric_df.columns[1],
                            z=numeric_df.columns[2] if numeric_df.shape[1] > 2 else None,
                            color="Cluster", title="KMeans Clustering Results")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 🧠 CNN Imaging (Chest X-Ray)
# ============================================================
elif app_mode == "CNN Imaging":
    st.header("🧠 CNN Imaging - Chest X-Ray Classification")
    uploaded = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels="BGR", caption="Uploaded Image", use_container_width=True)
        st.info("Pretend CNN model prediction: Pneumonia detected with 88.4% confidence ✅")

# ============================================================
# 🔗 ASSOCIATION RULES
# ============================================================
elif app_mode == "Association Rules":
    st.header("📚 Association Rule Mining")
    uploaded = st.file_uploader("Upload One-Hot Encoded CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        df = df.applymap(lambda x: 1 if str(x).lower() in ['true', '1', 'yes'] else 0)
        freq = apriori(df, min_support=0.2, use_colnames=True)
        rules = association_rules(freq, metric="lift", min_threshold=1.0)
        st.dataframe(rules)
        fig = px.scatter(rules, x="support", y="confidence", size="lift", hover_data=["antecedents", "consequents"])
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 🤖 CHATBOT
# ============================================================
elif app_mode == "Chatbot (AI Assistant)":
    healthcare_chatbot_component()

# ============================================================
# 🌍 TRANSLATOR
# ============================================================
elif app_mode == "Translator":
    from translator_app import translator_app
    translator_app()

# ============================================================
# 💬 SENTIMENT ANALYSIS
# ============================================================
elif app_mode == "Sentiment Analysis":
    from sentiment_app import sentiment_app
    sentiment_app()
