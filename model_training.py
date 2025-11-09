# ============================================

# model_training.py

# --------------------------------------------

# Loads preprocessed data, builds a pipeline,

# trains RandomForest with GridSearchCV,

# and saves the final model.

# ============================================

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

# Define directories

BASE_DIR = Path(**file**).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load training and validation data

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")

X_train = train_df.drop(columns=["readmit_30d"])
y_train = train_df["readmit_30d"]
X_val = val_df.drop(columns=["readmit_30d"])
y_val = val_df["readmit_30d"]

# Define feature categories

numeric_features = ["age", "length_of_stay", "num_prev_adm", "lab_glucose", "lab_creatinine", "followup_days"]
categorical_features = ["gender", "discharge_type", "smoker", "rural_resident", "chronic_diabetes", "chronic_hypertension"]

# Build preprocessing pipelines

numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])

preprocess = ColumnTransformer([
("num", numeric_transformer, numeric_features),
("cat", categorical_transformer, categorical_features)
])

# Create model pipeline

rf = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
pipe = Pipeline([("preprocess", preprocess), ("model", rf)])

# Grid search hyperparameter tuning

param_grid = {
"model__n_estimators": [150, 200, 300],
"model__max_depth": [None, 6, 10]
}

grid = GridSearchCV(pipe, param_grid, scoring="recall", cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print(f"[✔] Best Parameters: {grid.best_params_}")
print(f"[✔] Best Cross-Validation Recall: {grid.best_score_:.4f}")

# Save best model

best_model = grid.best_estimator_
model_path = MODEL_DIR / "readmission_pipeline.joblib"
joblib.dump(best_model, model_path)
print(f"[✔] Model saved to {model_path}")
