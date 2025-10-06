# main_dashboard.py
import os, io, time
import pandas as pd
import numpy as np
import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(page_title="HealthAI: End-to-End", layout="wide")

# Import local modules safely (catch syntax/import errors so app doesn't crash)
translator_ui = None
sentiment_ui = None
healthcare_chatbot_query = None
ensure_label_column = None
GENAI_READY = False

translator_import_error = None
sentiment_import_error = None
chatbot_import_error = None
prepare_import_error = None

try:
    from translator_app import translator_ui as _translator_ui
    translator_ui = _translator_ui
except Exception as e:
    translator_import_error = e

try:
    from sentiment_app import sentiment_ui as _sentiment_ui
    sentiment_ui = _sentiment_ui
except Exception as e:
    sentiment_import_error = e

try:
    from chatbot_frontend import healthcare_chatbot_query as _chatbot_q, GENAI_READY as _gready
    healthcare_chatbot_query = _chatbot_q
    GENAI_READY = bool(_gready)
except Exception as e:
    chatbot_import_error = e

try:
    from prepare_data import ensure_label_column as _ensure
    ensure_label_column = _ensure
except Exception as e:
    prepare_import_error = e

# Directory for data
DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)

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
st.sidebar.caption("Place CSVs / images under `data/` in the repo or upload via module pages.")

# HOME
if module == "Home":
    st.title("🏥 HealthAI: End-to-End AI/ML Healthcare Platform")
    st.markdown("""
    This dashboard demonstrates: classification, regression, clustering, association rules,
    imaging demo, chatbot (Gemini fallback), translator, and sentiment analysis.
    """)
    st.info("Ensure `requirements.txt` is installed. Put your data files in the `data/` folder.")

# CLASSIFICATION
elif module == "Classification (Disease Risk)":
    st.header("🧬 Disease Risk Classification")
    st.write("Upload a CSV with features and a `label` column (e.g., Low/Medium/High or 0/1).")
    uploaded = st.file_uploader("Upload classification CSV", type=["csv"], key="clf")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(100))
        if "label" not in df.columns:
            st.warning("'label' column is missing.")
            if ensure_label_column is not None and st.button("Create example label from `los`"):
                df = ensure_label_column(df, target_col="los")
                st.success("Label column added (example).")
        else:
            X = df.drop(columns=["label"]).select_dtypes(include=[np.number]).fillna(df.median())
            y = df["label"]
            if X.shape[1] == 0:
                st.error("No numeric features found. Provide numeric columns (age, bmi, bp...).")
            else:
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestClassifier
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                st.success(f"Test accuracy: {score:.3f}")
                importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
                import plotly.express as px
                fig = px.bar(importances.reset_index().rename(columns={"index":"feature",0:"importance"}),
                             x="feature", y=0, labels={"0":"importance"})
                st.plotly_chart(fig, use_container_width=True)

# REGRESSION
elif module == "Regression (LOS prediction)":
    st.header("📈 LOS Prediction (Regression)")
    st.write("Upload CSV with `los` numeric column for Length of Stay prediction.")
    uploaded = st.file_uploader("Upload regression CSV", type=["csv"], key="reg")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(100))
        if "los" not in df.columns:
            st.error("'los' column missing.")
        else:
            X = df.drop(columns=["los"]).select_dtypes(include=[np.number]).fillna(df.median())
            y = pd.to_numeric(df["los"], errors="coerce").fillna(0)
            if X.shape[1] == 0:
                st.error("No numeric features for regression.")
            else:
                from sklearn.linear_model import LinearRegression
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mae = np.mean(np.abs(preds - y_test))
                st.success(f"MAE: {mae:.3f}")
                import plotly.express as px
                fig = px.scatter(x=y_test, y=preds, labels={"x":"True LOS","y":"Predicted LOS"})
                st.plotly_chart(fig, use_container_width=True)

# CLUSTERING
elif module == "Clustering (Patient Segmentation)":
    st.header("🔎 Patient Segmentation (KMeans)")
    uploaded = st.file_uploader("Upload CSV for clustering", type=["csv"], key="clust")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(100))
        X = df.select_dtypes(include=[np.number]).fillna(df.median())
        if X.shape[1] == 0:
            st.error("No numeric columns found.")
        else:
            from sklearn.cluster import KMeans
            n = st.slider("Number of clusters", 2, 8, 3)
            k = KMeans(n_clusters=n, random_state=42).fit(X)
            labels = k.labels_
            st.success(f"Assigned {n} clusters.")
            try:
                from sklearn.decomposition import PCA
                import plotly.express as px
                proj = PCA(n_components=2, random_state=42).fit_transform(X)
                pdf = pd.DataFrame(proj, columns=["pc1","pc2"]); pdf["cluster"]=labels
                fig = px.scatter(pdf, x="pc1", y="pc2", color="cluster", title="Cluster projection (PCA)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.write("PCA visualization not available (missing dependency).")

# ASSOCIATION RULES
elif module == "Association Rule Mining":
    st.header("📚 Apriori Association Rules")
    uploaded = st.file_uploader("Upload one-hot transactions CSV (0/1 or True/False)", type=["csv"], key="assoc")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(100))
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            # ensure boolean
            df_bool = df.astype(bool)
            freq = apriori(df_bool, min_support=0.1, use_colnames=True)
            rules = association_rules(freq, metric="lift", min_threshold=1.0)
            st.write("Frequent itemsets:")
            st.dataframe(freq.sort_values("support", ascending=False).head(20))
            st.write("Association rules:")
            st.dataframe(rules.sort_values("lift", ascending=False).head(20))
        except Exception as e:
            st.error("Apriori/association_rules not available or input not one-hot: " + str(e))

# CNN IMAGING DEMO
elif module == "CNN (Imaging Demo)":
    st.header("🖼️ Imaging (CNN Demo)")
    uploaded = st.file_uploader("Upload an X-ray image (jpg/png)", type=["jpg","jpeg","png"], key="img")
    if uploaded:
        img_bytes = uploaded.read()
        st.image(img_bytes, caption="Uploaded image", use_column_width=False, width=350)
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((224,224))
        arr = np.array(img)/255.0
        score = arr.mean()
        pred = "PNEUMONIA" if score > 0.5 else "NORMAL"
        st.success(f"Demo prediction: {pred} (mock)")

# CHATBOT
elif module == "Chatbot (Gemini)":
    st.header("💬 Chatbot (Gemini)")
    st.write("Type your health question (no voice). If Gemini isn't configured, a fallback answer appears.")
    q = st.text_input("Ask a question", key="chat")
    if st.button("Ask"):
        if not q.strip():
            st.warning("Enter a question.")
        else:
            if healthcare_chatbot_query is None:
                st.error("Chatbot module not available. Import error: " + str(chatbot_import_error))
                st.write("Fallback: For health queries, consult clinicians. Try keywords like 'heart', 'diabetes' for simple canned answers.")
            else:
                try:
                    resp = healthcare_chatbot_query(q)
                    st.markdown(resp)
                    # feedback logging
                    rating = st.radio("Was this helpful?", ("", "Yes", "No"))
                    if rating:
                        os.makedirs(os.path.join(DATA_DIR, "feedback"), exist_ok=True)
                        flog = os.path.join(DATA_DIR, "feedback", "feedback_log.csv")
                        entry = {"timestamp": time.time(), "question": q, "answer": resp, "rating": rating}
                        pd.DataFrame([entry]).to_csv(flog, mode=("a" if os.path.exists(flog) else "w"), index=False, header=(not os.path.exists(flog)))
                        st.success("Feedback saved.")
                except Exception as e:
                    st.error("Chatbot error: " + str(e))

# TRANSLATOR
elif module == "Translator":
    st.header("🌐 Translator")
    if translator_ui is None:
        st.error("Translator module not available. Import error: " + str(translator_import_error))
    else:
        translator_ui()

# SENTIMENT
elif module == "Sentiment Analysis":
    st.header("🧾 Sentiment Analysis")
    if sentiment_ui is None:
        st.error("Sentiment module not available. Import error: " + str(sentiment_import_error))
    else:
        sentiment_ui()

# footer
st.markdown("---")
st.caption("HealthAI demo — put datasets under data/ and add GEMINI_API_KEY to Streamlit Secrets for Gemini access.")
