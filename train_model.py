
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    average_precision_score
)


# 1. Load dataset
df = pd.read_csv("data/fake_job_postings.csv")

print("Dataset loaded:", df.shape)


# 2. Select useful text columns
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "industry"
]


# 3. Handle missing values
df[text_columns] = df[text_columns].fillna("")


# 4. Combine text features
df["combined_text"] = df[text_columns].agg(" ".join, axis=1)

X = df["combined_text"]
y = df["fraudulent"]


# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 6. Create ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2
        )
    ),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    )
])


# 7. Train model
print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


# 8. Predictions
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# 9. Evaluation
print("\n========== MODEL EVALUATION ==========")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Genuine", "Fraudulent"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nPR-AUC (Average Precision):")
print(average_precision_score(y_test, y_prob))


# 10. Save model
joblib.dump(model, "detector/fraud_model.joblib")

print("\nModel saved successfully!")
