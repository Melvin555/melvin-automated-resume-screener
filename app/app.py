import streamlit as st
import joblib
import re

# Sidebar information
st.sidebar.title("About This App")
st.sidebar.markdown(
    """
    **Automated Resume Screener** uses Natural Language Processing (NLP) to classify resumes into job categories.

    ### What can be categorized?
    - Software roles (e.g., Python Developer, Java Developer, DevOps Engineer)
    - Data roles (e.g., Data Science, Database, ETL Developer)
    - Engineering (e.g., Mechanical Engineer, Electrical Engineering, Civil Engineer)
    - Business & Management (e.g., Business Analyst, Operations Manager, PMO)
    - Others (e.g., HR, Sales, Arts, Health and Fitness, Web Designing, Advocate, Blockchain, Testing, Automation Testing, SAP Developer, DotNet Developer, Hadoop, Network Security Engineer)

    ### Tips for better classification:
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

# Clean input text
def clean_input(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.lower()

st.title("🤖 Automated Resume Screener")
st.markdown("Paste your resume text below and click **Classify** to predict the job category.")

resume_text = st.text_area(
    "Paste Resume Text Here",
    height=350,
    placeholder="e.g.\nJohn Doe\nEmail: john.doe@email.com\n...\n"
)

if st.button("Classify"):
    cleaned = clean_input(resume_text)
    vect_input = vectorizer.transform([cleaned])
    prediction = model.predict(vect_input)[0]
    st.success(f"Predicted Job Category: **{prediction}**")
