# main_dashboard.py
import streamlit as st
st.set_page_config(page_title="HealthAI Dashboard", page_icon="🏥", layout="wide")

# standard imports
import pandas as pd
import numpy as np
import os
from PIL import Image
import plotly.express as px

# import helper modules (make sure they exist in repo)
from chatbot_frontend import healthcare_chatbot_query, GENAI_READY
from sentiment_app import sentiment_module_ui
from translator_app import translate_ui

st.title("🏥 HealthAI: End-to-End Healthcare AI/ML Dashboard")
st.write("A compact demo that follows your project deliverables: Classification, Regression (LOS), Clustering, Association rules, CNN inference, Chatbot, Translator, Sentiment.")

# show small sidebar for modules
st.sidebar.title("Select Module")
module = st.sidebar.radio("", ["Home", "Classification", "Regression", "Clustering", "Association Rules", "CNN Imaging", "Chatbot (AI Assistant)", "Translator", "Sentiment Analysis"])

# show assets safely
def load_asset_img(name, w=120):
    p = os.path.join("assets", name)
    if os.path.exists(p):
        try:
            return Image.open(p)
        except Exception:
            return None
    return None

img = load_asset_img("icon_chatbot.png")
if img:
    st.sidebar.image(img, width=120)

# HOME
if module == "Home":
    st.header("Welcome")
    st.markdown("""
    **This demo implements the core modules from your project brief.**
    - Small, fast classification/regression/clustering using CSVs in `/data/`
    - CNN: lightweight inference using MobileNetV2 on uploaded image
    - Chatbot: Gemini (preferred) or fallback short answers
    - Translator & Sentiment modules: light and fast
    """)
    st.info("Make sure your data files are placed in the `data/` folder in the repo (tabular CSVs, transactions.csv, images/).")

# CLASSIFICATION (simple)
elif module == "Classification":
    st.header("Classification (binary demo)")
    st.write("Upload a CSV with a binary `target` column and numeric features (age,bmi,bp).")
    up = st.file_uploader("Upload classification CSV", type="csv")
    if up:
        df = pd.read_csv(up)
        st.write("Columns:", df.columns.tolist())
        if 'target' not in df.columns:
            st.error("CSV must contain a `target` column (0/1).")
        else:
            # select numeric columns
            X = df.select_dtypes(include=[np.number]).drop(columns=['target'], errors=False)
            y = df['target']
            # simple train/test
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X.fillna(X.median()), y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            st.success(f"RandomForest test accuracy: {acc:.3f}")
            # feature importances
            fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.bar_chart(fi.head(10))
            st.write("Sample predictions:")
            preds = clf.predict(X_test.head(10))
            st.dataframe(pd.concat([X_test.head(10).reset_index(drop=True), pd.DataFrame({'pred':preds})], axis=1))

# REGRESSION (LOS)
elif module == "Regression":
    st.header("Regression: LOS prediction (simple demo)")
    st.write("Upload a CSV that contains `los` (length of stay) and numeric features.")
    up = st.file_uploader("Upload regression CSV", type="csv", key="reg")
    if up:
        df = pd.read_csv(up)
        if 'los' not in df.columns:
            st.error("CSV must include `los` column.")
        else:
            X = df.select_dtypes(include=[np.number]).drop(columns=['los'], errors=False).fillna(df.median())
            y = df['los'].fillna(df['los'].median())
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            reg = RandomForestRegressor(n_estimators=50, random_state=42)
            reg.fit(X_train, y_train)
            preds = reg.predict(X_test)
            from sklearn.metrics import mean_absolute_error, r2_score
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            st.write(f"MAE: {mae:.3f} — R2: {r2:.3f}")
            # quick scatter plot
            fig = px.scatter(x=y_test, y=preds, labels={'x':'True LOS','y':'Predicted LOS'}, title="True vs Predicted LOS")
            st.plotly_chart(fig, use_container_width=True)

# CLUSTERING
elif module == "Clustering":
    st.header("Patient Clustering (k-means)")
    up = st.file_uploader("Upload tabular CSV for clustering (numeric features)", type="csv", key="clus")
    if up:
        df = pd.read_csv(up)
        X = df.select_dtypes(include=[np.number]).fillna(df.median())
        from sklearn.cluster import KMeans
        k = st.slider("Number of clusters (k)", 2, 8, 3)
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        df['cluster'] = km.labels_
        st.write("Cluster counts:")
        st.write(df['cluster'].value_counts())
        # 2D PCA plot
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        vals = pca.fit_transform(X)
        fig = px.scatter(x=vals[:,0], y=vals[:,1], color=df['cluster'].astype(str), labels={'x':'PC1','y':'PC2'}, title="Cluster visualization (PCA 2D)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.head(50))

# ASSOCIATION RULES
elif module == "Association Rules":
    st.header("Association Rules (Apriori)")
    st.write("Upload a one-hot encoded CSV (columns are items with 0/1 indicating presence).")
    up = st.file_uploader("Upload transactions CSV (one-hot)", type="csv", key="assoc")
    if up:
        df = pd.read_csv(up)
        # The mlxtend apriori expects booleans or 0/1 numeric
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            bool_df = df.astype(bool)
            freq = apriori(bool_df, min_support=0.1, use_colnames=True)
            rules = association_rules(freq, metric="confidence", min_threshold=0.5)
            st.write("Frequent items:")
            st.dataframe(freq.sort_values(by='support', ascending=False).head(20))
            st.write("Derived rules (sample):")
            st.dataframe(rules.head(20))
        except Exception as e:
            st.error(f"Apriori failed: {e}. Make sure CSV is one-hot encoded (0/1 columns per item).")

# CNN Imaging (inference)
elif module == "CNN Imaging":
    st.header("CNN Imaging (Chest X-ray inference demo)")
    st.write("Upload a chest x-ray image (JPEG/PNG). We run a tiny MobileNetV2 classifier for demo only.")
    uploaded_file = st.file_uploader("Upload an image", type=['png','jpg','jpeg'])
    if uploaded_file:
        try:
            img = Image.open(uploaded_file).convert('RGB').resize((224,224))
            st.image(img, caption="Uploaded image", use_column_width=False, width=300)
            st.write("Running small inference (MobileNetV2 pretrained on imagenet — not medical).")
            import tensorflow as tf
            from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
            import numpy as np
            model = MobileNetV2(weights='imagenet')
            x = np.array(img).astype('float32')
            x = preprocess_input(x)
            x = np.expand_dims(x, 0)
            preds = model.predict(x)
            decoded = decode_predictions(preds, top=3)[0]
            st.write("Top predicted ImageNet classes (not medical):")
            for c in decoded:
                st.write(f"- {c[1]}  ({c[2]*100:.2f}%)")
        except Exception as e:
            st.error(f"Image inference error: {e}")

# Chatbot
elif module == "Chatbot (AI Assistant)":
    st.header("Healthcare Chatbot")
    st.write("Ask any health-related question in English or Tamil (the bot will attempt to answer in the same language). Short answers by default.")
    q = st.text_input("Ask your question (type or paste)")
    short = st.checkbox("Short answer (1-2 lines)", value=True)
    if st.button("Ask"):
        if not q:
            st.error("Please type a question.")
        else:
            with st.spinner("Getting answer..."):
                ans = healthcare_chatbot_query(q, short_answer=short)
                st.info(ans)

# Translator
elif module == "Translator":
    translate_ui()

# Sentiment
elif module == "Sentiment Analysis":
    sentiment_module_ui()

# Footer
st.markdown("---")
st.caption("Developed as a compact demo for HealthAI End-to-End project. Use Streamlit Secrets for GEMINI_API_KEY.")
