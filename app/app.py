import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load("models/resume_classifier.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

# Clean input text
def clean_input(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.lower()

st.title("🤖 Automated Resume Screener")
resume_text = st.text_area("Paste Resume Text Here")

if st.button("Classify"):
    cleaned = clean_input(resume_text)
    vect_input = vectorizer.transform([cleaned])
    prediction = model.predict(vect_input)[0]
    st.success(f"Predicted Job Category: **{prediction}**")
