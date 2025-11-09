# ============================================

# evaluation.py

# --------------------------------------------

# Evaluates the trained model on test data

# and prints accuracy, precision, recall.

# ============================================

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib
from pathlib import Path

# Directories

BASE_DIR = Path(**file**).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models" / "readmission_pipeline.joblib"

# Load model and test data

model = joblib.load(MODEL_PATH)
test_df = pd.read_csv(DATA_DIR / "test.csv")

X_test = test_df.drop(columns=["readmit_30d"])
y_test = test_df["readmit_30d"]

# Generate predictions

y_pred = model.predict(X_test)

# Evaluate model

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n[📊] Model Evaluation Results:")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(cm)
