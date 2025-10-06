import streamlit as st
import pandas as pd
import nltk
import google.generativeai as genai

# ----------------------------------------------------------
# ✅ STEP 1: Ensure NLTK + TextBlob dependencies work safely
# ----------------------------------------------------------
try:
            from textblob import TextBlob
            # Check for required corpora on Streamlit Cloud
            try:
                            nltk.data.find('tokenizers/punkt')
except LookupError:
                nltk.download('punkt')
                nltk.download('averaged_perceptron_tagger')
                nltk.download('wordnet')
                nltk.download('omw-1.4')

except ImportError:
            # Fallback if TextBlob import fails
            class TextBlob:
                            def __init__(self, text):
                                                self.text = text

                            @property
                            def sentiment(self):
                                                score = sum([1 if w in self.text.lower() else 0 for w in ["good", "excellent", "happy", "satisfied"]]) - \
                                                        sum([1 if w in self.text.lower() else 0 for w in ["bad", "poor", "sad", "angry", "terrible"]])
                                                return type("Sentiment", (), {"polarity": score / 10})


# ----------------------------------------------------------
# ✅ STEP 2: Configure Gemini API (set your API key in secrets)
# ----------------------------------------------------------
try:
            api_key = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-pro")
except Exception as e:
            st.warning("⚠️ Gemini API key missing or invalid. Please add GEMINI_API_KEY in Streamlit secrets.")
            model = None


# ----------------------------------------------------------
# ✅ STEP 3: Define Sentiment Module Function
# ----------------------------------------------------------
def sentiment_module_ui():
            st.markdown("## 💬 Sentiment Analysis on Patient Feedback")
            st.markdown("Analyze what patients feel about hospital services, doctors, or treatments.")

    uploaded_file = st.file_uploader("📤 Upload patient feedback CSV (with a 'feedback' column)", type=["csv"])
    user_feedback = st.text_area("📝 Or enter feedback manually", placeholder="e.g., The staff were very caring and helpful.")

    if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                    if "feedback" not in df.columns:
                                        st.error("❌ The CSV must contain a 'feedback' column.")
                                        return
                                    st.success("✅ File uploaded successfully!")

        # Compute sentiment
        df["polarity"] = df["feedback"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df["Sentiment"] = df["polarity"].apply(lambda x: "Positive 😀" if x > 0 else ("Negative 😞" if x < 0 else "Neutral 😐"))

        st.dataframe(df[["feedback", "Sentiment"]])

        # Plot sentiment distribution
        st.bar_chart(df["Sentiment"].value_counts())

        if model:
                            st.subheader("🧠 Gemini Insights:")
                            try:
                                                    summary_text = "\n".join(df["feedback"].astype(str).tolist())
                                                    prompt = f"Summarize the following hospital patient feedback and identify key improvement areas:\n{summary_text}"
                                                    gemini_summary = model.generate_content(prompt)
                                                    st.info(gemini_summary.text)
except Exception as e:
                st.warning("⚠️ Could not generate Gemini insights. Please check API usage or input size.")

elif user_feedback:
        # Manual feedback input
        blob = TextBlob(user_feedback)
        polarity = blob.sentiment.polarity

        if polarity > 0:
                            sentiment = "Positive 😀"
elif polarity < 0:
            sentiment = "Negative 😞"
else:
            sentiment = "Neutral 😐"

        st.success(f"**Detected Sentiment:** {sentiment}")

        if model:
                            try:
                                                    prompt = f"Analyze this patient feedback and suggest an empathetic hospital response:\n'{user_feedback}'"
                                                    reply = model.generate_content(prompt)
                                                    st.info(reply.text)
except Exception:
                st.warning("⚠️ Gemini could not respond. Try again later.")


# ----------------------------------------------------------
# ✅ STEP 4: Allow Standalone Execution
# ----------------------------------------------------------
if __name__ == "__main__":
            st.set_page_config(page_title="HealthAI Sentiment Analysis", page_icon="💬")
    sentiment_module_ui()
