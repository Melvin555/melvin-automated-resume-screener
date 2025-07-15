# 🤖 Automated Resume Screener

This project uses Natural Language Processing (NLP) to automatically classify resumes into job categories using a Logistic Regression model and TF-IDF features.

## 🚀 Features

- Resume text classification
- **Two ways to predict:**  
  - Paste resume text directly into the app  
  - Upload one or more resume files (PDF or DOCX) for batch classification
- Interactive web app with Streamlit
- Download batch prediction results as CSV
- Displays prediction confidence for each result
- Trained on real resume data

## 🛠 Tech Stack

- Python
- scikit-learn
- Streamlit
- TF-IDF Vectorizer
- Logistic Regression

## 📊 Dataset

Public Kaggle dataset: [Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)

## ▶️ Run the App

```sh
git clone git@github.com:YourName/automated-resume-screener.git
cd automated-resume-screener
conda create -n resume-nlp python=3.10
conda activate resume-nlp
pip install -r requirements.txt
streamlit run app/app.py
```

## 🏋️ Model Training & Evaluation

After running `train_model.py`, the model is trained on the cleaned and deduplicated resume dataset. During training, the following steps and results are observed:

- **Duplicates removed:** 796 duplicate resumes were found and removed for better generalization.
- **Class imbalance warning:** Some classes have fewer than 5 samples (e.g., Operations Manager, Web Designing, PMO). This can make predictions for these classes less reliable.

- **Classification report:**  
  The model's performance on the test set is summarized below:

    ```
    precision    recall  f1-score   support

    Advocate                1.00      1.00      1.00         2
    Arts                    1.00      1.00      1.00         1
    Automation Testing      0.33      1.00      0.50         1
    Blockchain              1.00      1.00      1.00         1
    Business Analyst        0.50      1.00      0.67         1
    Civil Engineer          1.00      1.00      1.00         1
    Data Science            1.00      1.00      1.00         2
    Database                1.00      1.00      1.00         2
    DevOps Engineer         0.00      0.00      0.00         1
    DotNet Developer        0.00      0.00      0.00         2
    ETL Developer           1.00      1.00      1.00         1
    Electrical Engineering  1.00      1.00      1.00         1
    HR                      1.00      1.00      1.00         2
    Hadoop                  1.00      1.00      1.00         2
    Health and fitness      1.00      1.00      1.00         1
    Java Developer          0.75      1.00      0.86         3
    Mechanical Engineer     0.00      0.00      0.00         1
    Network Security Eng.   1.00      1.00      1.00         1
    Operations Manager      0.50      1.00      0.67         1
    PMO                     0.00      0.00      0.00         1
    Python Developer        0.25      1.00      0.40         1
    SAP Developer           1.00      1.00      1.00         1
    Sales                   1.00      1.00      1.00         1
    Testing                 0.00      0.00      0.00         2
    Web Designing           0.00      0.00      0.00         1

    accuracy                           0.76        34
    macro avg       0.65      0.76      0.68        34
    weighted avg    0.67      0.76      0.70        34
    ```

- **Cross-validated accuracy:**  
  `0.8316 ± 0.0553` (mean ± std), indicating the model performs reasonably well overall, but performance may vary across classes.

- **Sanity check (shuffled labels):**  
  Accuracy drops to `0.0419` when labels are shuffled, confirming the model is learning meaningful patterns and there is no data leakage.

### 📉 Confusion Matrix

The confusion matrix visualizes the model's predictions versus the true categories on the test set. This helps identify which classes are most often confused with each other.

![Confusion Matrix](figure/training_log.png)

**How to interpret:**  
- Each row represents the true class, and each column represents the predicted class.
- Diagonal values (top-left to bottom-right) show correct predictions.
- Off-diagonal values indicate misclassifications.
- Sparse or empty rows/columns for some classes (especially those with few samples) indicate the model struggles to predict or encounter those classes.

**Note:**  
Classes with very few samples (less than 5) may have unreliable metrics and may not be well represented in the confusion matrix.

---

## 📝 Example: Classifying a Resume

### Step-by-step

1. **Start the app** using the command above.

2. **Choose your input method:**
   - **Paste Resume Text:**  
     Copy and paste your resume text into the "Paste Resume Text Here" input box.
   - **Upload Resume Files:**  
     Upload one or more resumes in **PDF** or **DOCX** format using the file uploader.  
     - You can upload a single file for individual prediction, or multiple files for batch prediction.
     - The app will extract the text and classify each resume automatically.

3. **Click the appropriate button:**
   - For pasted text, click **"Classify Pasted Text"**.
   - For uploaded files, the app will display batch results automatically.

4. **View the result:**  
   - For single predictions, the app will display the predicted job category and the model's confidence.
   - For batch predictions, a table will show the filename, predicted category, and confidence for each file.
   - You can also **download the batch results as a CSV file**.

5. **Example UI Output:**  
   ![Prediction UI Example](figure/prediction_ui.png)

---

## 📂 What Can Be Uploaded?

- **Accepted formats:** PDF (`.pdf`) and Word Document (`.docx`)
- **Single or multiple files:** Upload one or many resumes at once for batch processing.
- **Content:** The resumes should contain relevant information such as:
  - Name and contact info (optional)
  - Professional summary
  - Skills (e.g., programming languages, frameworks, tools)
  - Work experience (job titles, companies, years)
  - Education and certifications

The more detailed and relevant your resume content, the more accurate the classification will be.

---

This demonstrates how to use the app to classify a resume by pasting text or uploading files, and what kind of output to expect.