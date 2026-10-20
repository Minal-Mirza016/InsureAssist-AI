from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_DIR / "training" / "intent_dataset.csv"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

df = pd.read_csv(DATASET_PATH)

# Remove any rows with missing text or labels
df = df.dropna(subset=["text", "intent"])

# Remove duplicate questions
df = df.drop_duplicates(subset=["text"])

# Clean whitespace
df["text"] = df["text"].str.strip()
df["intent"] = df["intent"].str.strip()

print("Dataset shape:", df.shape)
print("\nExamples per intent:")
print(df["intent"].value_counts())


# --------------------------------------------------
# 3. Separate inputs and labels
# --------------------------------------------------

X = df["text"]
y = df["intent"]

# X = questions supplied to the model
# y = correct intent label for each question


# --------------------------------------------------
# 4. Split into training and testing sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining examples:", len(X_train))
print("Testing examples:", len(X_test))


# --------------------------------------------------
# 5. Build the ML pipeline
# --------------------------------------------------

model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)


# --------------------------------------------------
# 6. Train the model
# --------------------------------------------------

model.fit(X_train, y_train)

print("\nModel training complete!")


# --------------------------------------------------
# 7. Evaluate on unseen test examples
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nCLASSIFICATION REPORT")
report = classification_report(
    y_test,
    y_pred,
    zero_division=0,
)
print(report)

# Save the report
report_path = RESULTS_DIR / "classification_report.txt"
report_path.write_text(
    f"Test Accuracy: {accuracy:.4f}\n\n{report}",
    encoding="utf-8",
)


# --------------------------------------------------
# 8. Generate a confusion matrix
# --------------------------------------------------

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels,
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels,
)

fig, ax = plt.subplots(figsize=(12, 10))

display.plot(
    ax=ax,
    xticks_rotation=45,
    colorbar=False,
)

plt.title("Insurance Intent Classifier - Confusion Matrix")
plt.tight_layout()

matrix_path = RESULTS_DIR / "confusion_matrix.png"
plt.savefig(matrix_path, dpi=150)
plt.close(fig)


# --------------------------------------------------
# 9. Save the trained model
# --------------------------------------------------

model_path = MODEL_DIR / "intent_pipeline.joblib"

joblib.dump(model, model_path)

print("\nSaved model to:", model_path)
print("Saved report to:", report_path)
print("Saved confusion matrix to:", matrix_path)