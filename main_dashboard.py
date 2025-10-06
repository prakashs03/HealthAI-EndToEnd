# main_dashboard.py
# Streamlit main dashboard for HealthAI End-to-End project
# IMPORTANT: set streamlit run main_dashboard.py or set this as main in Streamlit Cloud

import os
import io
import time
import pandas as pd
import numpy as np
import streamlit as st

# PAGE CONFIG (must be first streamlit call)
st.set_page_config(page_title="HealthAI: End-to-End", layout="wide")

# Local imports (your other modules)
from chatbot_frontend import healthcare_chatbot_query, GENAI_READY
from translator_app import translator_ui
from sentiment_app import sentiment_ui
from prepare_data import ensure_label_column

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px

# IMPORTANT paths (adjust if your repo structure differs)
DATA_DIR = os.path.join(os.getcwd(), "data")

# Sidebar
st.sidebar.title("📁 Modules")
module = st.sidebar.radio("Choose a module", [
    "Home",
    "Classification (Disease Risk)",
    "Regression (LOS prediction)",
    "Clustering (Patient Segmentation)",
    "Association Rule Mining",
    "CNN (Imaging Demo)",
    "Chatbot (Gemini)",
    "Translator",
    "Sentiment Analysis"
])

st.sidebar.markdown("---")
st.sidebar.markdown("Data directory: `data/`")
st.sidebar.markdown("Make sure uploaded CSVs match the expected columns. See README for details.")

# --- HOME ---
if module == "Home":
    st.title("🏥 HealthAI: End-to-End AI/ML Healthcare Platform")
    st.markdown("""
    Welcome to the HealthAI dashboard — an end-to-end platform for ML on clinical datasets.
    Modules included: Classification, Regression, Clustering, Association Rules, CNN demo, Chatbot, Translator, Sentiment.
    """)
    st.info("Place your CSV/images under the `data/` folder in the repo or upload via module pages.")

# --- CLASSIFICATION ---
elif module == "Classification (Disease Risk)":
    st.title("🧬 Disease Risk Classification")
    st.markdown("Upload a CSV with features and a `label` column (values: 0/1 or Low/Medium/High).")
    uploaded = st.file_uploader("Upload classification CSV", type=["csv"], key="clf")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(200))
        if "label" not in df.columns:
            st.error("⚠️ 'label' column missing! You can run prepare_data to add a label.")
            if st.button("Create a simple binary label (example)"):
                df = ensure_label_column(df, target_col="los")  # fallback: create label from LOS if present
                st.success("Label column added.")
        else:
            # Basic preprocessing
            X = df.drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number]).fillna(0)
            y = df["label"]
            if X.shape[1] == 0:
                st.error("No numeric features were found. Please provide numeric features (age, bmi, bp, etc.)")
            else:
                st.info(f"Training RandomForest on {X.shape[0]} samples, {X.shape[1]} numeric features...")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                st.success(f"Test accuracy: {score:.3f}")
                importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                fig = px.bar(importances.reset_index().rename(columns={"index":"feature",0:"importance"}), x="feature", y=0,
                             title="Feature Importances")
                st.plotly_chart(fig, use_container_width=True)

# --- REGRESSION ---
elif module == "Regression (LOS prediction)":
    st.title("📈 Length of Stay (LOS) Prediction")
    st.markdown("Upload CSV with numeric features and an `los` column (numeric).")
    uploaded = st.file_uploader("Upload regression CSV", type=["csv"], key="reg")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(200))
        if "los" not in df.columns:
            st.error("⚠️ 'los' column missing!")
        else:
            X = df.drop(columns=["los"]).select_dtypes(include=[np.number]).fillna(df.median())
            y = df["los"].astype(float)
            if X.shape[1] == 0:
                st.error("No numeric features found for regression.")
            else:
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mae = np.mean(np.abs(preds - y_test))
                st.success(f"MAE on test: {mae:.3f}")
                fig = px.scatter(x=y_test, y=preds, labels={"x":"True LOS", "y":"Predicted LOS"}, title="True vs Predicted LOS")
                st.plotly_chart(fig, use_container_width=True)

# --- CLUSTERING ---
elif module == "Clustering (Patient Segmentation)":
    st.title("🔎 Patient Segmentation (KMeans)")
    uploaded = st.file_uploader("Upload tabular CSV for clustering", type=["csv"], key="clust")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(200))
        X = df.select_dtypes(include=[np.number]).fillna(df.median())
        n_clusters = st.slider("K (clusters)", 2, 10, 3)
        if X.shape[1] == 0:
            st.error("No numeric columns for clustering.")
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X)
            st.success(f"Assigned {n_clusters} clusters.")
            df_vis = X.copy()
            df_vis["cluster"] = labels
            # two-D PCA for visualization (quick)
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=42)
                proj = pca.fit_transform(X)
                vis_df = pd.DataFrame(proj, columns=["pc1","pc2"])
                vis_df["cluster"] = labels
                fig = px.scatter(vis_df, x="pc1", y="pc2", color="cluster", title="Cluster projection (PCA 2D)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.write("Could not run PCA visualization (missing dependency).")

# --- ASSOCIATION RULE MINING ---
elif module == "Association Rule Mining":
    st.title("📚 Apriori Association Rules (one-hot CSV)")
    uploaded = st.file_uploader("Upload one-hot encoded transactions CSV (columns as items, values 0/1/True/False)", type=["csv"], key="assoc")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(200))
        try:
            # Validate values are 0/1 or True/False
            ok = True
            vals = set(np.unique(df.values))
            allowed = {0,1,True,False, "0","1","True","False"}
            if not vals.issubset(allowed):
                st.error("Error: the dataframe must be one-hot encoded (values 0/1 or True/False). Found other values.")
            else:
                freq = apriori(df.astype(bool), min_support=0.1, use_colnames=True)
                rules = association_rules(freq, metric="lift", min_threshold=1.0)
                st.write("Top 10 frequent itemsets:")
                st.dataframe(freq.sort_values("support", ascending=False).head(10))
                st.write("Top 10 rules by lift:")
                st.dataframe(rules.sort_values("lift", ascending=False).head(10))
        except Exception as e:
            st.error("Association mining failed: " + str(e))

# --- CNN IMAGING (demo) ---
elif module == "CNN (Imaging Demo)":
    st.title("🖼️ CNN Imaging (Demo)")
    st.markdown("Upload a chest X-ray image (JPG/PNG). Demo inference (mock) will be returned.")
    uploaded = st.file_uploader("Upload X-ray image", type=["jpg","jpeg","png"], key="image")
    if uploaded is not None:
        img_bytes = uploaded.read()
        st.image(img_bytes, caption="Uploaded X-ray", use_column_width=False, width=350)
        # Demo: simple grayscale mean heuristic as mock model
        import PIL.Image as Image
        img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((224,224))
        arr = np.array(img)/255.0
        score = arr.mean()
        pred = "PNEUMONIA" if score > 0.5 else "NORMAL"
        st.success(f"Demo CNN ready — predicted: **{pred}** (mock prediction)")

# --- CHATBOT ---
elif module == "Chatbot (Gemini)":
    st.title("💬 Healthcare Chatbot (Gemini)")
    st.markdown("Type your health-related question and get answers. (No voice.)")
    question = st.text_input("Ask your health question:", key="chat_q")
    if st.button("Ask") and question.strip() != "":
        with st.spinner("Querying chatbot..."):
            try:
                resp = healthcare_chatbot_query(question)
                st.markdown(resp)
                # auto-log feedback placeholder (user can provide rating)
                rating = st.radio("Was this answer helpful?", ("", "Yes", "No"))
                if rating:
                    # append to feedback log
                    os.makedirs(os.path.join(DATA_DIR, "feedback"), exist_ok=True)
                    flog = os.path.join(DATA_DIR, "feedback", "feedback_log.csv")
                    entry = {"timestamp": time.time(), "question": question, "answer": resp, "rating": rating}
                    df_log = pd.DataFrame([entry])
                    if os.path.exists(flog):
                        df_log.to_csv(flog, mode="a", index=False, header=False)
                    else:
                        df_log.to_csv(flog, index=False)
                    st.success("Thanks — feedback logged.")
            except Exception as e:
                st.error("Chatbot error: " + str(e))
    else:
        st.info("Ask a question and press Ask.")

# --- TRANSLATOR ---
elif module == "Translator":
    st.title("🌐 Translator")
    translator_ui()

# --- SENTIMENT ANALYSIS ---
elif module == "Sentiment Analysis":
    st.title("🧾 Sentiment Analysis")
    sentiment_ui()

# Footer
st.markdown("---")
st.caption("HealthAI — End-to-End AI/ML healthcare demo. Follow project README for dataset and deployment instructions.")
