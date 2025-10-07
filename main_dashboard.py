# main_dashboard.py
# Main Streamlit dashboard that ties modules together.
import os
import io
import time
import tempfile
import warnings

warnings.filterwarnings("ignore")

try:
    import streamlit as st
except Exception as e:
    raise RuntimeError("Please install streamlit. pip install streamlit") from e

# Standard data science libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# Imaging
from PIL import Image

# local modules
from chatbot_frontend import healthcare_chatbot_query, GENIE_AVAILABLE
from translator_app import translate_text
from sentiment_app import analyze_sentiment, log_feedback

# App config
st.set_page_config(page_title="HealthAI End-to-End", layout="wide")

# --- Helpers ---
DATA_DIR = "data"
MODELS_DIR = "models"
SAVED_MODELS_DIR = "saved_models"

def ensure_data_path():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

ensure_data_path()

def ensure_label_column(df: pd.DataFrame, target_col="los"):
    """Add a simple label column for demo classification if missing.
    If target_col numeric (e.g., los) convert to bins Low/Medium/High.
    """
    if "label" in df.columns:
        return df
    if target_col in df.columns:
        # bin into 3 equal-frequency buckets
        df = df.copy()
        df["label"] = pd.qcut(df[target_col].fillna(df[target_col].median()), q=3, labels=["Low","Medium","High"])
        return df
    # else try to create simple synthetic label from other features
    df = df.copy()
    if "age" in df.columns and "bmi" in df.columns:
        score = (df["age"].fillna(50)/100.0) + (df["bmi"].fillna(25)/50.0)
        df["label"] = pd.cut(score, bins=3, labels=["Low","Medium","High"])
        return df
    # fallback: random label (deterministic seed)
    df["label"] = pd.Series(np.random.RandomState(42).choice(["Low","Medium","High"], size=len(df)))
    return df

def to_numeric_df(X: pd.DataFrame):
    # select numeric columns only
    return X.select_dtypes(include=[np.number])

def quick_plot_hist(df, col):
    fig, ax = plt.subplots()
    sns.histplot(df[col].dropna(), ax=ax, kde=True)
    st.pyplot(fig)

# --- UI Layout ---
st.sidebar.title("Select Module")
module = st.sidebar.radio("Choose an Analysis Module", [
    "Home",
    "Classification",
    "Regression",
    "Clustering",
    "Association Rules",
    "CNN Imaging",
    "LSTM (Time Series)",
    "Chatbot (Gemini)",
    "Translator",
    "Sentiment Analysis"
])

st.title("HealthAI: End-to-End AI/ML Healthcare Platform")

if module == "Home":
    st.markdown("""
    Welcome to **HealthAI Dashboard** — an end-to-end demonstration of healthcare analytics:
    - Classification (disease risk)
    - Regression (length of stay)
    - Clustering (patient cohorts)
    - Association rule mining (comorbidities)
    - Imaging (CNN)
    - Time-series forecasting (LSTM)
    - NLP modules: Chatbot, Translator, Sentiment
    """)
    st.markdown("**Data folder**: ensure your CSVs and images are in `data/` folder in repo.")
    st.markdown("**Models folder**: pre-trained models go into `models/` (optional).")

# -------------------------
# 1) Classification module
# -------------------------
if module == "Classification":
    st.header("🧬 Disease Risk Classification")
    st.info("Upload classification CSV (must have a 'label' or 'los' column).")
    uploaded = st.file_uploader("Upload classification CSV", type=["csv"], key="clf_csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(200))
        df = ensure_label_column(df, target_col="los")
        if "label" not in df.columns:
            st.error("'label' column missing after attempt to generate. Provide a CSV with 'label' column.")
        else:
            # Preprocess
            X = df.drop(columns=["label"])
            X_num = to_numeric_df(X).fillna(X.median())
            if X_num.shape[1] == 0:
                st.error("No numeric features found. Provide numeric columns (age, bmi, bp, glucose, cholesterol).")
            else:
                y = LabelEncoder().fit_transform(df["label"].astype(str))
                st.write("Features used:", list(X_num.columns))
                X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                report = classification_report(y_test, preds, output_dict=True)
                st.subheader("Classification report")
                st.table(pd.DataFrame(report).transpose())
                # feature importances
                importances = pd.Series(model.feature_importances_, index=X_num.columns).sort_values(ascending=False)
                st.subheader("Feature importances")
                st.bar_chart(importances)
                st.success("Classification demo complete.")

# -------------------------
# 2) Regression module
# -------------------------
if module == "Regression":
    st.header("📈 Length of Stay (LOS) Prediction - Regression")
    st.info("Upload regression CSV with a numeric 'los' column and numeric features (age,bmi,bp).")
    uploaded = st.file_uploader("Upload regression CSV", type=["csv"], key="reg_csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(200))
        if "los" not in df.columns:
            st.error("'los' column missing! The regression module expects a numeric 'los' column.")
        else:
            X = df.drop(columns=["los"])
            X_num = to_numeric_df(X).fillna(X.median())
            if X_num.shape[1] == 0:
                st.error("No numeric features found. Provide numeric columns (age, bmi, bp, glucose, cholesterol).")
            else:
                y = df["los"].astype(float).fillna(df["los"].median())
                X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=42)
                reg = RandomForestRegressor(n_estimators=100, random_state=42)
                reg.fit(X_train, y_train)
                preds = reg.predict(X_test)
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                st.write(f"MAE: {mae:.2f} , R2: {r2:.3f}")
                # plot actual vs predicted
                fig, ax = plt.subplots()
                ax.scatter(y_test, preds, alpha=0.6)
                ax.set_xlabel("Actual LOS")
                ax.set_ylabel("Predicted LOS")
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
                st.pyplot(fig)
                st.success("Regression demo complete.")

# -------------------------
# 3) Clustering
# -------------------------
if module == "Clustering":
    st.header("🔎 Patient Segmentation (Clustering)")
    uploaded = st.file_uploader("Upload tabular CSV (numeric features)", type=["csv"], key="cluster_csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(200))
        X_num = to_numeric_df(df).fillna(df.median())
        if X_num.shape[1] == 0:
            st.error("No numeric columns found for clustering.")
        else:
            n_clusters = st.slider("Number of clusters", 2, 8, 3)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X_num)
            km = KMeans(n_clusters=n_clusters, random_state=42)
            labels = km.fit_predict(Xs)
            df["cluster"] = labels
            st.subheader("Cluster counts")
            st.write(df["cluster"].value_counts())
            # PCA for 2D plot
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                XY = pca.fit_transform(Xs)
                fig, ax = plt.subplots()
                scatter = ax.scatter(XY[:,0], XY[:,1], c=labels, cmap="tab10", alpha=0.7)
                ax.set_title("Clustering (PCA projection)")
                st.pyplot(fig)
            except Exception:
                st.info("PCA not available - skip 2D projection.")
            st.success("Clustering done. Download cluster assignment:")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download labeled data", data=csv, file_name="clustered_patients.csv")

# -------------------------
# 4) Association Rules
# -------------------------
if module == "Association Rules":
    st.header("📚 Association Rule Mining (Apriori)")
    st.info("Upload one-hot encoded transactions CSV (columns True/False or 0/1).")
    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"], key="assoc_csv")
    if uploaded:
        trans = pd.read_csv(uploaded)
        st.dataframe(trans.head(200))
        # Validate: values must be boolean or 0/1
        def validate_onehot(df):
            ok = True
            for c in df.columns:
                vals = pd.unique(df[c].dropna())
                allowed = set([0,1,True,False,"0","1"])
                if not set(vals).issubset(allowed):
                    ok = False
                    break
            return ok

        if not validate_onehot(trans):
            st.error("Allowed values for a one-hot dataframe are True/False or 0/1. Convert your dataset first.")
        else:
            # convert to boolean 0/1
            trans_bool = trans.copy().astype(int)
            freq = apriori(trans_bool, min_support=0.1, use_colnames=True)
            rules = association_rules(freq, metric="lift", min_threshold=1.0)
            st.subheader("Frequent itemsets")
            st.dataframe(freq.sort_values("support", ascending=False).head(50))
            st.subheader("Top association rules")
            st.dataframe(rules.sort_values("lift", ascending=False).head(50))
            st.success("Association mining complete.")

# -------------------------
# 5) CNN Imaging
# -------------------------
if module == "CNN Imaging":
    st.header("🖼 CNN Imaging Diagnostics (Chest X-ray)")
    st.info("Upload an X-ray (jpg/png). A local model in models/cnn_best.h5 will be used if present.")
    img_file = st.file_uploader("Upload X-ray", type=["jpg","png","jpeg"], key="xray")
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, caption="Uploaded X-Ray", width=400)
        model_path = os.path.join(MODELS_DIR, "cnn_best.h5")
        if os.path.exists(model_path):
            st.success("Found model cnn_best.h5 - running prediction")
            try:
                import tensorflow as tf
                from tensorflow.keras.models import load_model
                m = load_model(model_path)
                # simple preprocessing
                x = img.resize((224,224))
                arr = np.array(x)/255.0
                arr = arr[np.newaxis,...]
                pred = m.predict(arr)
                # assume binary: Pneumonia vs Normal
                if pred.shape[-1] == 1 or pred.shape[-1] == 2:
                    score = float(pred[0].max())
                    cls = "PNEUMONIA" if pred[0].argmax()==1 else "NORMAL"
                else:
                    score = float(np.max(pred))
                    cls = "PNEUMONIA" if np.argmax(pred)==1 else "NORMAL"
                st.success(f"Predicted: {cls} (confidence {score:.2f})")
            except Exception as e:
                st.error("Error loading/using CNN model. See logs.")
                st.exception(e)
        else:
            st.info("No model found. Showing demo/mock prediction.")
            # mock logic: random based on simple brightness
            arr = np.array(img.convert("L")).astype(float)
            mean_brightness = arr.mean()
            if mean_brightness < 100:
                st.success("Predicted: PNEUMONIA (demo)")
            else:
                st.success("Predicted: NORMAL (demo)")
            st.warning("To use real CNN prediction, place a trained Keras model at models/cnn_best.h5")

# -------------------------
# 6) LSTM Forecasting
# -------------------------
if module == "LSTM (Time Series)":
    st.header("⏱ LSTM (Vital Signs Forecasting) - demo")
    st.info("Upload vitals.csv with columns: timestamp, hr, spo2, bp_systolic, bp_diastolic")
    uploaded = st.file_uploader("Upload vitals CSV", type=["csv"], key="vitals_csv")
    if uploaded:
        df = pd.read_csv(uploaded, parse_dates=True)
        st.dataframe(df.head(200))
        st.info("This module demonstrates time-window based forecasting. For full LSTM training, use notebook.")
        if "hr" in df.columns:
            st.line_chart(df[["hr"]].fillna(method="ffill"))
        else:
            st.error("No hr column found. Provide vitals with numeric columns")

# -------------------------
# 7) Chatbot (Gemini)
# -------------------------
if module == "Chatbot (Gemini)":
    st.header("💬 Healthcare Chatbot (Gemini if available)")
    st.info("Type your health question. The bot will return 1-2 line answers by default. Click 'Explain' for full answer.")
    query = st.text_input("Ask a health-related question:", key="chat_query")
    if query:
        short = st.checkbox("Short answer (1-2 lines)", value=True)
        explain = st.button("Explain (detailed)")
        # call wrapper
        answer = healthcare_chatbot_query(query, short=short, explain=explain)
        st.markdown(answer)
        # log feedback
        feedback = st.text_area("If you want to leave feedback on the answer (optional):")
        if st.button("Submit feedback"):
            log_feedback(query, answer, feedback)
            st.success("Feedback logged")

# -------------------------
# 8) Translator
# -------------------------
if module == "Translator":
    st.header("🌐 Translator (English <-> Tamil demo)")
    txt = st.text_area("Text to translate (English or Tamil):")
    if st.button("Translate"):
        out = translate_text(txt)
        st.write(out)

# -------------------------
# 9) Sentiment Analysis
# -------------------------
if module == "Sentiment Analysis":
    st.header("❤️ Sentiment Analysis (Patient Feedback)")
    st.info("Upload patient_feedback.csv (csv with a 'text' column).")
    uploaded = st.file_uploader("Upload feedback CSV", type=["csv"], key="sent_csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(200))
        if "text" not in df.columns:
            st.error("CSV must have a 'text' column.")
        else:
            df["sentiment"] = df["text"].apply(lambda t: analyze_sentiment(str(t)))
            st.dataframe(df[["text","sentiment"]].head(200))
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results", data=csv, file_name="feedback_with_sentiment.csv")

# Footer
st.markdown("---")
st.markdown("Developed for End-to-End HealthAI project. Place your datasets in `data/` and models in `models/`.")
