# ============================================

# data_preprocessing.py

# --------------------------------------------

# Generates synthetic hospital data, performs

# train/validation/test split, and saves files.

# ============================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Setup data paths

BASE_DIR = Path(**file**).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Generate synthetic patient data

rng = np.random.default_rng(42)
n_samples = 5000

age = rng.integers(18, 90, size=n_samples)
gender = rng.choice(["Male", "Female"], size=n_samples)
length_of_stay = rng.integers(1, 21, size=n_samples)
num_prev_adm = rng.poisson(0.8, size=n_samples)
chronic_diabetes = rng.integers(0, 2, size=n_samples)
chronic_hypertension = rng.integers(0, 2, size=n_samples)
lab_glucose = rng.normal(110, 25, size=n_samples)
lab_creatinine = rng.normal(1.0, 0.3, size=n_samples)
discharge_type = rng.choice(
["Home", "Rehab", "SkilledNursing", "AgainstAdvice"],
size=n_samples, p=[0.7, 0.15, 0.12, 0.03]
)
followup_days = rng.integers(3, 30, size=n_samples)
smoker = rng.integers(0, 2, size=n_samples)
rural_resident = rng.integers(0, 2, size=n_samples)

# Introduce missing values in lab results

mask_missing = rng.random(n_samples) < 0.05
lab_glucose[mask_missing] = np.nan

# Generate readmission probabilities

logit = (
0.02 * (age - 50)
+ 0.12 * length_of_stay
+ 0.35 * num_prev_adm
+ 0.4 * chronic_diabetes
+ 0.3 * chronic_hypertension
+ 0.01 * (np.nan_to_num(lab_glucose, nan=110) - 110)
+ 0.6 * (np.nan_to_num(lab_creatinine, nan=1.0) - 1.0)
+ 0.2 * smoker
- 0.03 * followup_days
+ 0.15 * rural_resident
+ rng.normal(0, 0.5, size=n_samples)
)

# Adjust readmission probability by discharge type

dt_effect = {"Home": 0.0, "Rehab": 0.25, "SkilledNursing": 0.35, "AgainstAdvice": 0.8}
logit += np.vectorize(dt_effect.get)(discharge_type)

# Convert to binary class (0 = No, 1 = Yes)

prob = 1 / (1 + np.exp(-logit))
y = (prob > 0.7).astype(int)

# Create DataFrame

df = pd.DataFrame({
"age": age, "gender": gender, "length_of_stay": length_of_stay,
"num_prev_adm": num_prev_adm, "chronic_diabetes": chronic_diabetes,
"chronic_hypertension": chronic_hypertension, "lab_glucose": lab_glucose,
"lab_creatinine": lab_creatinine, "discharge_type": discharge_type,
"followup_days": followup_days, "smoker": smoker,
"rural_resident": rural_resident, "readmit_30d": y
})

# Save raw data

raw_path = RAW_DIR / "patient_readmission_synthetic.csv"
df.to_csv(raw_path, index=False)
print(f"[✔] Synthetic dataset saved to {raw_path}")

# Split data into train, validation, and test

X = df.drop(columns=["readmit_30d"])
y = df["readmit_30d"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Save splits

X_train.assign(readmit_30d=y_train).to_csv(PROCESSED_DIR / "train.csv", index=False)
X_val.assign(readmit_30d=y_val).to_csv(PROCESSED_DIR / "val.csv", index=False)
X_test.assign(readmit_30d=y_test).to_csv(PROCESSED_DIR / "test.csv", index=False)

print("[✔] Data split into train/validation/test sets and saved.")
