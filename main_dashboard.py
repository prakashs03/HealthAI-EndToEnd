# main_dashboard.py
# Main Streamlit dashboard that ties together modules you already have.
# Place this file at repository root. Make sure your folder structure matches:
# - models/
# - data/
# - chatbot_frontend.py
# - sentiment_app.py
# - translator_app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path

# --- Page config MUST be first Streamlit command (we keep it first) ---
st.set_page_config(page_title="HealthAI End-to-End", layout="wide", initial_sidebar_state="expanded")

# Attempt to import other modules (they contain gemini wrappers + fallbacks)
try:
    from chatbot_frontend import healthcare_chatbot_query
except Exception as e:
    healthcare_chatbot_query = None
    st.warning("Chatbot module import failed — chatbot will show fallback responses.")

try:
    from sentiment_app import sentiment_module_ui, analyze_sentiment
except Exception as e:
    sentiment_module_ui = None
    analyze_sentiment = None
    st.warning("Sentiment module import failed — sentiment UI disabled.")

try:
    from translator_app import translator_ui, translate_text
except Exception as e:
    translator_ui = None
    translate_text = None
    st.warning("Translator module import failed — translator UI disabled.")

# Helper paths
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# Sidebar - Modules
st.sidebar.title("Modules")
module = st.sidebar.radio("Choose a module", (
    "Home",
    "Classification (Disease Risk)",
    "Regression (LOS prediction)",
    "Clustering (Patient segmentation)",
    "Association Rule Mining",
    "CNN (Imaging Diagnostics)",
    "LSTM (Time Series)",
    "Chatbot (Gemini)",
    "Translator",
    "Sentiment Analysis"
))

# ---------------------- Home ----------------------
if module == "Home":
    st.title("HealthAI: End-to-End AI/ML Healthcare Platform")
    st.write("Welcome — platform demonstrator for classification, regression, clustering, association rules, CNN imaging, LSTM forecasting, Translator, Chatbot (Gemini) and Sentiment analysis.")
    st.write("Make sure to add `GEMINI_API_KEY` in Streamlit Secrets for Gemini-powered modules.")
    st.markdown("**Data & Models:** `data/` and `models/` folders should be present in the repo root.")

# ---------------------- Classification ----------------------
elif module == "Classification (Disease Risk)":
    st.header("Disease Risk Classification")
    st.write("Upload a CSV with features and a `label` column (or click 'Generate label' to create an example label).")
    uploaded = st.file_uploader("Upload classification CSV (with 'label' column)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(10))
        # check label
        if "label" not in df.columns:
            if st.button("Generate example label column (adds 'label')"):
                # simple binning demo on 'bmi' or 'age' if present
                if "bmi" in df.columns:
                    df["label"] = pd.cut(df["bmi"], bins=[0,18.5,25,40,999], labels=["Low","Medium","High","Very High"]).astype(str)
                elif "age" in df.columns:
                    df["label"] = pd.cut(df["age"], bins=[0,30,50,150], labels=["Low","Medium","High"]).astype(str)
                else:
                    df["label"] = np.random.choice(["Low","Medium","High"], size=len(df))
                st.success("Generated 'label' column (download below).")
                st.download_button("Download labeled CSV", df.to_csv(index=False), "labeled_classification.csv")
        else:
            st.success("'label' column detected — ready to train or run models.")
            # Quick baseline: show class distribution
            st.write("Label distribution:")
            st.bar_chart(df["label"].value_counts())

            # Basic ML: load saved model if exists
            clf_path = MODELS_DIR / "tabular_clf.joblib"
            if clf_path.exists():
                model = joblib.load(clf_path)
                st.write("Loaded pre-trained tabular classifier from models/tabular_clf.joblib")
            else:
                st.info("No saved classifier found in models/; show baseline metrics or training instructions.")

# ---------------------- Regression ----------------------
elif module == "Regression (LOS prediction)":
    st.header("Length of Stay (LOS) Prediction — Regression")
    st.write("Upload CSV with numeric features and `los` column (target).")
    uploaded = st.file_uploader("Upload regression CSV (must include 'los')", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(10))
        if "los" not in df.columns:
            st.error("'los' column missing! Please upload a file with a numeric 'los' column.")
        else:
            # keep numeric cols
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] <= 1:
                st.error("Need numeric features other than the target 'los'.")
            else:
                st.write("Numeric features preview:")
                st.dataframe(numeric.head(5))
                # If saved model exists, load and predict
                los_model_path = MODELS_DIR / "los_reg.joblib"
                if los_model_path.exists():
                    los_model = joblib.load(los_model_path)
                    st.success("Loaded LOS regression model (models/los_reg.joblib)")
                    X = numeric.drop(columns=["los"], errors="ignore")
                    if X.shape[1] == 0:
                        st.error("No features left after dropping 'los'.")
                    else:
                        preds = los_model.predict(X.fillna(X.median()))
                        st.metric("Mean Absolute Error", float(np.mean(np.abs(preds - df["los"]))))
                        st.line_chart(pd.DataFrame({"true": df["los"], "pred": preds}).reset_index(drop=True))
                else:
                    st.info("No saved LOS model in models/; you can train locally and save as models/los_reg.joblib")

# ---------------------- Clustering ----------------------
elif module == "Clustering (Patient segmentation)":
    st.header("Patient Segmentation (Clustering)")
    st.write("Upload a patient CSV (numeric features). Will run KMeans baseline.")
    uploaded = st.file_uploader("Upload dataset for clustering (CSV)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(8))
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 1:
            st.error("No numeric columns detected for clustering.")
        else:
            from sklearn.cluster import KMeans
            k = st.slider("Number of clusters k", min_value=2, max_value=10, value=3)
            km = KMeans(n_clusters=k, random_state=42).fit(numeric.fillna(numeric.median()))
            df["cluster"] = km.labels_
            st.write("Cluster counts:")
            st.bar_chart(df["cluster"].value_counts())
            st.dataframe(df.head(10))

# ---------------------- Association Rules ----------------------
elif module == "Association Rule Mining":
    st.header("Apriori Association Rule Mining")
    st.write("Upload one-hot encoded transactions CSV (columns with 0/1 or True/False).")
    uploaded = st.file_uploader("Upload one-hot encoded CSV (transactions)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(10))
        # validation: only boolean / 0/1 allowed
        try:
            # convert possible True/False to numeric
            df_bool = df.copy()
            for c in df_bool.columns:
                if df_bool[c].dtype == object:
                    # try to map
                    df_bool[c] = df_bool[c].map({True:1, False:0, "True":1, "False":0, "true":1, "false":0}).fillna(df_bool[c])
            # require only 0/1 now
            invalid = df_bool.applymap(lambda x: x not in (0,1)).any().any()
            if invalid:
                st.error("The allowed values for a DataFrame are True, False, 0, 1. Please convert the file to one-hot binary format.")
            else:
                from mlxtend.frequent_patterns import apriori, association_rules
                freq = apriori(df_bool.astype(int), min_support=0.2, use_colnames=True)
                rules = association_rules(freq, metric="confidence", min_threshold=0.6)
                st.write("Top association rules:")
                st.dataframe(rules.sort_values("lift", ascending=False).head(10))
        except Exception as e:
            st.error("Error processing for apriori. Ensure file is one-hot encoded binary per column.")

# ---------------------- CNN Imaging ----------------------
elif module == "CNN (Imaging Diagnostics)":
    st.header("CNN Imaging Diagnostics (Chest X-ray)")
    st.write("Upload an X-ray image (JPG/PNG). The app will use models/cnn_best.h5 if available.")
    uploaded = st.file_uploader("Upload X-ray image", type=["jpg","jpeg","png"])
    if uploaded:
        from PIL import Image
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded X-Ray", use_column_width=False, width=350)
        cnn_path = MODELS_DIR / "cnn_best.h5"
        if cnn_path.exists():
            st.success("Loaded cnn_best.h5 — performing prediction")
            # quick preprocessing
            import numpy as np
            from tensorflow.keras.preprocessing.image import img_to_array
            model = tf.keras.models.load_model(str(cnn_path))
            arr = img.resize((224,224))
            x = img_to_array(arr)/255.0
            x = np.expand_dims(x,0)
            pred = model.predict(x)
            # expecting binary classification demo
            label = "PNEUMONIA" if pred[0].argmax() == 1 else "NORMAL"
            st.success(f"Prediction: {label}")
        else:
            st.info("No CNN model found in models/cnn_best.h5 — this page shows demo image only.")

# ---------------------- LSTM Time Series ----------------------
elif module == "LSTM (Time Series)":
    st.header("LSTM (Vitals forecasting / readmission)")
    st.write("Upload CSV of time-series vitals (rows ordered chronologically). Example expects columns like HR, RR, BP.")
    uploaded = st.file_uploader("Upload time-series CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(10))
        lstm_path = MODELS_DIR / "lstm_stub.h5"
        if lstm_path.exists():
            st.success("Loaded LSTM stub model.")
            st.info("This demo shows how to plug real LSTM — model usage depends on model input shape.")
        else:
            st.info("No LSTM model found in models/ — show basic time-series plot instead.")
            st.line_chart(df.select_dtypes(include=[np.number]).fillna(0).head(200))

# ---------------------- Chatbot ----------------------
elif module == "Chatbot (Gemini)":
    st.header("Healthcare Chatbot (Gemini)")
    st.write("Ask a health-related question. By default the bot replies with 1-2 line answers. Add 'explain' to get details.")
    prompt = st.text_input("Ask your health question (type then press Enter):")
    if prompt:
        if healthcare_chatbot_query is None:
            st.error("Chatbot module not available on import.")
        else:
            # call chatbot wrapper
            try:
                reply = healthcare_chatbot_query(prompt)
                st.info(reply)
            except Exception as e:
                st.error("Chatbot error: " + str(e))

# ---------------------- Translator ----------------------
elif module == "Translator":
    st.header("Translator (Gemini-powered fallback)")
    if translator_ui is None:
        st.error("Translator module not available.")
    else:
        translator_ui()

# ---------------------- Sentiment Analysis ----------------------
elif module == "Sentiment Analysis":
    st.header("Sentiment Analysis (Patient Feedback)")
    if sentiment_module_ui is None:
        st.error("Sentiment module not available.")
    else:
        sentiment_module_ui()

# ---------------------- End ----------------------
else:
    st.write("Select a module.")
