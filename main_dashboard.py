# main_dashboard.py
import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, mean_absolute_error
from pathlib import Path

# local modules
from chatbot_frontend import healthcare_chatbot_component
from sentiment_app import sentiment_component
from translator_app import translator_component

st.set_page_config(page_title="Gemini-Powered Healthcare AI", layout="wide")

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SAVED_DIR = ROOT / "saved_models"
SAVED_DIR.mkdir(exist_ok=True)

st.title("Gemini-Powered Healthcare AI/ML Pipeline")

menu = st.sidebar.radio("Select Module", [
    "Home", "Classification", "Regression", "Clustering",
    "Association Rules", "CNN Imaging", "LSTM Forecasting",
    "Chatbot (AI Assistant)", "Translator", "Sentiment Analysis"
])

###########################
# Home
###########################
if menu == "Home":
    st.markdown("""
    ### Project: End-to-End Healthcare AI
    Modules available:
    - Classification (risk stratification)
    - Regression (length-of-stay)
    - Clustering (patient cohorts)
    - Association Rules (comorbidities)
    - Imaging (CNN skeleton using chest X-rays)
    - Time-series (LSTM skeleton)
    - NLP: Chatbot (Gemini), Translator, Sentiment
    """)
    st.info("Place datasets inside the `data/` folder. Images in `data/images/train` and `data/images/test` with `NORMAL` and `PNEUMONIA` subfolders.")

###########################
# Helper to load tabular
###########################
def load_tabular(name="tabular_complete.csv"):
    path = DATA_DIR / name
    if not path.exists():
        st.warning(f"Missing {name} in data/ — please upload or check path.")
        return None
    df = pd.read_csv(path)
    return df

###########################
# Classification
###########################
if menu == "Classification":
    st.header("Classification — Disease Risk")
    df = load_tabular("tabular_complete.csv")
    if df is not None:
        st.write("Preview dataset (head):")
        st.dataframe(df.head())

        # pick target if exists
        if "target" not in df.columns:
            st.error("Expected 'target' column in tabular_complete.csv for classification.")
        else:
            # choose features
            X = df.drop(columns=["target"])
            # numeric only
            X = X.select_dtypes(include=[np.number]).fillna(df.median())
            y = df["target"]
            st.write(f"Using numeric features: {list(X.columns)}")

            test_size = st.slider("Test size", 0.1, 0.5, 0.2)
            random_state = 42
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
            with st.spinner("Training RandomForest..."):
                clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.json(report)

            st.subheader("Feature importances")
            fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.bar_chart(fi.head(10))

###########################
# Regression (LOS)
###########################
if menu == "Regression":
    st.header("Regression — Length of Stay (LOS)")
    df = load_tabular("tabular_complete.csv")
    if df is not None:
        st.dataframe(df.head())
        if "los" not in df.columns:
            st.error("Expected 'los' column in tabular_complete.csv for regression.")
        else:
            X = df.drop(columns=["los"])
            X = X.select_dtypes(include=[np.number]).fillna(df.median())
            y = df["los"]
            st.write(f"Numeric features used: {list(X.columns)}")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            reg = RandomForestRegressor(n_estimators=100, random_state=42)
            with st.spinner("Training regressor..."):
                reg.fit(X_train, y_train)
            preds = reg.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            st.write(f"MAE: {mae:.3f}")

            fig = px.scatter(x=y_test, y=preds, labels={"x":"Actual LOS","y":"Predicted LOS"})
            st.plotly_chart(fig, use_container_width=True)

###########################
# Clustering
###########################
if menu == "Clustering":
    st.header("Clustering — KMeans patient cohorting")
    df = load_tabular("tabular_complete.csv")
    if df is not None:
        X = df.select_dtypes(include=[np.number]).fillna(df.median())
        n_clusters = st.slider("n_clusters", 2, 8, 4)
        km = KMeans(n_clusters=n_clusters, random_state=42)
        with st.spinner("Fitting KMeans..."):
            labels = km.fit_predict(X)
        st.write("Cluster counts:")
        st.write(pd.Series(labels).value_counts())
        # 2D scatter of first two features
        if X.shape[1] >= 2:
            fig = px.scatter(x=X.iloc[:,0], y=X.iloc[:,1], color=labels, labels={"x":X.columns[0],"y":X.columns[1]})
            st.plotly_chart(fig, use_container_width=True)

###########################
# Association Rules (simple check using one-hot transactional CSV)
###########################
if menu == "Association Rules":
    st.header("Association Rules")
    tpath = DATA_DIR / "transactions.csv"
    if not tpath.exists():
        st.warning("Upload one-hot encoded transactions.csv into data/ for association mining.")
    else:
        df_transactions = pd.read_csv(tpath)
        st.write(df_transactions.head())
        try:
            # lazy run of mlxtend apriori (if installed)
            from mlxtend.frequent_patterns import apriori, association_rules
            min_support = st.slider("min_support", 0.05, 0.5, 0.2)
            with st.spinner("Running Apriori..."):
                freq = apriori(df_transactions, min_support=min_support, use_colnames=True)
                rules = association_rules(freq, metric="confidence", min_threshold=0.3)
            st.dataframe(rules.sort_values("lift", ascending=False).head(20))
        except Exception as e:
            st.error(f"Association mining requires mlxtend. Error: {e}")

###########################
# CNN Imaging (skeleton, quick demo)
###########################
if menu == "CNN Imaging":
    st.header("CNN Imaging — Chest X-ray demo (skeleton)")
    st.write("Images should be placed at data/images/train/NORMAL, data/images/train/PNEUMONIA etc.")
    train_dir = DATA_DIR / "images" / "train"
    test_dir = DATA_DIR / "images" / "test"
    if not train_dir.exists():
        st.warning("No images found — upload train/test folders under data/images/")
    else:
        # Show sample images
        sample_class = st.selectbox("Show sample class", options=[p.name for p in train_dir.iterdir() if p.is_dir()])
        sample_folder = train_dir / sample_class
        sample_images = list(sample_folder.glob("*"))
        if sample_images:
            st.image(str(sample_images[0]), caption=f"{sample_class} sample", width=300)
        st.info("Full CNN training is heavy; this page is a demo placeholder. You can point to a pre-trained model in saved_models/ and run inference.")

###########################
# LSTM Forecasting (skeleton)
###########################
if menu == "LSTM Forecasting":
    st.header("LSTM Time-series Forecast (skeleton)")
    st.write("Use vitals.csv in data/ to run a short LSTM demo in notebook (heavy for Streamlit).")

###########################
# Chatbot
###########################
if menu == "Chatbot (AI Assistant)":
    st.header("Chatbot (AI Assistant)")
    healthcare_chatbot_component(root=ROOT)

###########################
# Translator
###########################
if menu == "Translator":
    st.header("Translator (EN <-> TA)")
    translator_component(root=ROOT)

###########################
# Sentiment
###########################
if menu == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    sentiment_component(root=ROOT)

# footer
st.markdown("---")
st.caption("Developed as part of End-to-End Healthcare AI project. Module outputs are demo-quality and meant for demonstration and evaluation. Always consult clinical experts before production use.")
