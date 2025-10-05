# ==========================================
# chatbot_frontend.py — Streamlit UI
# ==========================================
import streamlit as st
import requests
import time

st.set_page_config(page_title="Gemini HealthBot", page_icon="💬", layout="wide")
st.title("🤖 Gemini HealthBot — Smart Healthcare Assistant")

API_URL = "http://127.0.0.1:8000/ask"  # Replace with Streamlit Cloud API URL when deployed

user_input = st.text_input("💭 Ask me anything about your health:")

if st.button("Send"):
    if user_input.strip():
        with st.spinner("Analyzing your question..."):
            try:
                response = requests.post(API_URL, json={"question": user_input})
                data = response.json()
                st.success(data.get("answer", "⚠️ No response"))
            except Exception as e:
                st.error(f"⚠️ Request failed: {e}")
    else:
        st.warning("Please enter a question before sending.")
