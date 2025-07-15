import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np

# 1. Load cleaned data
df = pd.read_csv("data/resumes_cleaned.csv")  # Make sure the path matches your file

# 1a. Remove duplicate resumes
num_duplicates = df.duplicated(subset=["cleaned_resume"]).sum()
if num_duplicates > 0:
    print(f"⚠️ Found {num_duplicates} duplicate resumes. Removing duplicates for better generalization.")
    df = df.drop_duplicates(subset=["cleaned_resume"])

# 1b. Warn if any class has very few samples
class_counts = df["Category"].value_counts()
few_samples = class_counts[class_counts < 5]
if not few_samples.empty:
    print("⚠️ Warning: The following classes have fewer than 5 samples:")
    print(few_samples)

X = df["cleaned_resume"]
y = df["Category"]

# 2. Split BEFORE vectorization to prevent data leakage
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. TF-IDF vectorization (fit only on train)
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)

# 4. Train classifier with class balancing
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# 5. Evaluate model
y_pred = model.predict(X_test)
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# 6. Confusion matrix plot
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(xticks_rotation=90)
plt.tight_layout()
plt.show()

# 7. Optional: cross-validated accuracy
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
print(f"\n✅ Cross-validated accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 8. Sanity check: shuffle labels and check accuracy
y_shuffled = np.random.permutation(y)
shuffled_scores = cross_val_score(pipeline, X, y_shuffled, cv=cv, scoring="accuracy")
print(f"\n🧪 Sanity check (shuffled labels) accuracy: {shuffled_scores.mean():.4f}")
if shuffled_scores.mean() > 0.2:
    print("⚠️ Warning: High accuracy with shuffled labels may indicate data leakage or overfitting.")

# 9. Save model and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/resume_classifier.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

print("\n✅ Model and vectorizer saved in /models")
