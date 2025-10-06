# main_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from mlxtend.frequent_patterns import apriori, association_rules

# local module imports (chatbot, translator, sentiment)
from chatbot_frontend import healthcare_chatbot_component
from translator_app import translator_ui
from sentiment_app import sentiment_module_ui

st.set_page_config(page_title="HealthAI Dashboard", layout="wide")
st.title("🏥 HealthAI: End-to-End Healthcare Platform")

st.markdown("""
This dashboard demonstrates the modules required by the project:
1. Classification (Risk), 2. Regression (LOS), 3. Clustering, 4. Association Rules,
5. CNN (demo), 6. LSTM (demo), 7. Chatbot, 8. Translator, 9. Sentiment.
""")

menu = st.sidebar.selectbox("Choose module", [
    "Classification", "Regression", "Clustering", "Association Rules",
    "CNN Demo", "LSTM Demo", "Chatbot", "Translator", "Sentiment"
])

# helper: safe numeric extraction
def numeric_df(df):
    num = df.select_dtypes(include=[np.number])
    return num.fillna(num.median())

# 1. Classification
if menu == "Classification":
    st.subheader("Classification (Risk Stratification)")
    st.write("Upload a CSV with numeric features and a 'label' column (0/1).")
    f = st.file_uploader("CSV for classification", type="csv")
    if f:
        df = pd.read_csv(f)
        st.dataframe(df.head())
        if "label" not in df.columns:
            st.error("CSV must contain 'label' column. Use prepare_synthetic_data.py to generate sample.")
        else:
            X = numeric_df(df)
            y = df["label"]
            # simple model metrics using sklearn (no long training)
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            st.success(f"Test accuracy: {acc:.3f}")
            # show feature importances if available
            try:
                importances = model.feature_importances_
                feat = X.columns
                fig = px.bar(x=feat, y=importances, title="Feature Importances")
                st.plotly_chart(fig)
            except Exception:
                pass

# 2. Regression
elif menu == "Regression":
    st.subheader("Regression (Length of Stay)")
    st.write("Upload CSV containing numeric features and 'los' column.")
    f = st.file_uploader("CSV for regression", type="csv")
    if f:
        df = pd.read_csv(f)
        st.dataframe(df.head())
        if "los" not in df.columns:
            st.error("CSV must contain 'los' column.")
        else:
            X = numeric_df(df).drop(columns=["los"], errors="ignore")
            y = pd.to_numeric(df["los"], errors="coerce").fillna(0)
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            if X.shape[1] == 0:
                st.error("No numeric features found.")
            else:
                model.fit(X, y)
                preds = model.predict(X)
                fig = px.scatter(x=y, y=preds, labels={'x':'Actual LOS','y':'Predicted LOS'}, title="Actual vs Predicted LOS")
                st.plotly_chart(fig)
                mae = np.mean(np.abs(y - preds))
                st.write(f"MAE: {mae:.3f}")

# 3. Clustering
elif menu == "Clustering":
    st.subheader("Clustering (Patient Segmentation)")
    f = st.file_uploader("CSV for clustering", type="csv")
    if f:
        df = pd.read_csv(f)
        st.dataframe(df.head())
        X = numeric_df(df)
        if X.shape[1] < 2:
            st.error("Please upload CSV with at least 2 numeric features.")
        else:
            from sklearn.cluster import KMeans
            n_clusters = st.slider("Clusters", 2, 6, 3)
            km = KMeans(n_clusters=n_clusters, random_state=42)
            df['cluster'] = km.fit_predict(X)
            st.success("Clustering done.")
            fig = px.scatter(df, x=X.columns[0], y=X.columns[1], color='cluster', title="Clusters")
            st.plotly_chart(fig)

# 4. Association Rules
elif menu == "Association Rules":
    st.subheader("Association Rules (Apriori)")
    st.write("Upload one-hot encoded CSV (each column symptom/comorbidity, rows=transactions).")
    f = st.file_uploader("One-hot encoded CSV", type="csv")
    if f:
        df = pd.read_csv(f)
        try:
            # ensure boolean
            df_bool = df.fillna(0).astype(bool)
            freq = apriori(df_bool, min_support=0.15, use_colnames=True)
            rules = association_rules(freq, metric="lift", min_threshold=1.1)
            st.dataframe(rules[['antecedents','consequents','support','confidence','lift']].head(20))
        except Exception as e:
            st.error(f"Error computing rules: {e}")

# 5. CNN demo (no heavy training)
elif menu == "CNN Demo":
    st.subheader("CNN Demo (Chest X-rays)")
    st.write("Place images in `data/images/train/{NORMAL,PNEUMONIA}` and `data/images/test/{NORMAL,PNEUMONIA}`")
    train_dir = "data/images/train"
    test_dir = "data/images/test"
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        # show counts
        def count_classes(base):
            counts = {}
            for cls in ["NORMAL", "PNEUMONIA"]:
                p = os.path.join(base, cls)
                counts[cls] = len([f for f in os.listdir(p)]) if os.path.isdir(p) else 0
            return counts
        st.write("Train counts:", count_classes(train_dir))
        st.write("Test counts:", count_classes(test_dir))
        st.info("A small demo CNN will be shown (compiled but not trained here to keep it fast).")
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        model = Sequential([
            Conv2D(16,(3,3),activation='relu',input_shape=(64,64,3)),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        st.write("Demo CNN compiled. Use local training script if you want to train fully.")
    else:
        st.error("data/images/train & data/images/test not found. Upload images to those folders in repo.")

# 6. LSTM demo
elif menu == "LSTM Demo":
    st.subheader("LSTM Demo (Vitals time-series)")
    st.write("This is a demo visualization using synthetic vitals.")
    t = np.arange(0,200)
    signal = np.sin(t/10) + np.random.normal(0,0.1,size=len(t))
    fig = px.line(x=t, y=signal, title="Simulated heart-rate-like vitals")
    st.plotly_chart(fig)

# 7. Chatbot
elif menu == "Chatbot":
    st.subheader("Chatbot (Gemini if configured, otherwise fallback)")
    healthcare_chatbot_component()

# 8. Translator
elif menu == "Translator":
    st.subheader("Translator")
    translator_ui()

# 9. Sentiment
elif menu == "Sentiment":
    st.subheader("Sentiment")
    sentiment_module_ui()

st.markdown("---")
st.caption("Project: End-to-end HealthAI | Modules: Classification, Regression, Clustering, Association, CNN (demo), LSTM (demo), Chatbot (Gemini), Translator, Sentiment.")
