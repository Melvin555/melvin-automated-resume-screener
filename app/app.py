import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import joblib
import re
from utils.file_extractor import extract_text_from_pdf, extract_text_from_docx
from utils.predictor import predict_with_confidence, batch_predict

# Branding and layout
col1, col2 = st.columns([1, 3])
with col1:
    st.image("figure/app_logo.jpg", width=120)
with col2:
    st.title("🤖 Automated Resume Screener")
    st.markdown("Paste your resume text below, upload a file, or upload multiple files for batch classification.")

# Sidebar information
st.sidebar.title("About This App")
st.sidebar.markdown(
    """
    **Automated Resume Screener** uses NLP to classify resumes into job categories.

    ### ✨ Latest Features
    - **Paste resume text** for instant prediction.
    - **Batch prediction:** Upload one or more resumes as PDF or DOCX files for automatic classification.
    - **Download results:** Export batch predictions as a CSV file.
    - **Prediction confidence:** See how confident the model is for each prediction.

    ### 📂 What can be uploaded?
    - **Accepted formats:** PDF (`.pdf`) and Word Document (`.docx`)
    - **Single or multiple files:** Upload one or many resumes at once for batch processing.
    - **Content:** The resumes should contain relevant information such as:
        - Name and contact info (optional)
        - Professional summary
        - Skills (e.g., programming languages, frameworks, tools)
        - Work experience (job titles, companies, years)
        - Education and certifications

    ### 🏷️ What can be categorized?
    - Software roles (Python Developer, Java Developer, DevOps Engineer, etc.)
    - Data roles (Data Science, Database, ETL Developer)
    - Engineering (Mechanical, Electrical, Civil)
    - Business & Management (Business Analyst, Operations Manager, PMO)
    - Others (HR, Sales, Arts, Health and Fitness, Web Designing, Advocate, Blockchain, Testing, SAP Developer, DotNet Developer, Hadoop, Network Security Engineer)

    ### 💡 Tips for better classification:
    - **Include:** Skills, work experience, education, certifications, and technologies/tools used.
    - **Be specific:** Mention programming languages, frameworks, and job titles.
    - **Example input:**
        ```
        Jane Smith
        Email: jane.smith@email.com

        Skills: Python, Django, REST APIs, SQL, Data Analysis
        Experience: Python Developer at TechCorp (2021-2024)
        Education: B.Sc. Computer Science
        ```
    - The more detailed and relevant your resume text, the more accurate the prediction.
    """
)

# Load model and vectorizer
model = joblib.load("models/resume_classifier.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

def clean_input(text):
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text.lower()

# File upload section
st.subheader("Upload Resume(s) (PDF or DOCX)")
uploaded_files = st.file_uploader(
    "Choose one or more resume files",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

# Single text input section
st.subheader("Or Paste Resume Text")
resume_text = st.text_area(
    "Paste Resume Text Here",
    height=350,
    placeholder="e.g.\nJohn Doe\nEmail: john.doe@email.com\n...\n"
)

# Process single text input
if st.button("Classify Pasted Text"):
    if resume_text.strip():
        cleaned = clean_input(resume_text)
        pred, proba = predict_with_confidence(model, vectorizer, cleaned)
        st.success(f"Predicted Job Category: **{pred}** (Confidence: {proba:.2f})")
    else:
        st.warning("Please paste some resume text.")

# Process uploaded files (batch or single)
if uploaded_files:
    texts = []
    filenames = []
    for file in uploaded_files:
        if file.name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file)
        elif file.name.lower().endswith(".docx"):
            text = extract_text_from_docx(file)
        else:
            text = ""
        cleaned = clean_input(text)
        texts.append(cleaned)
        filenames.append(file.name)
    if texts:
        st.write("### Batch Prediction Results")
        batch_results = batch_predict(model, vectorizer, texts)
        batch_results.insert(0, "Filename", filenames)
        st.dataframe(batch_results)
        csv = batch_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name='resume_predictions.csv',
            mime='text/csv',
        )