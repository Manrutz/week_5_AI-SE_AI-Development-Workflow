# ============================================

# deployment_simulation.py

# --------------------------------------------

# Loads trained model and simulates predictions

# on new patient data for deployment testing.

# ============================================

import pandas as pd
import joblib
from pathlib import Path

# Load trained model

BASE_DIR = Path(**file**).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "readmission_pipeline.joblib"
model = joblib.load(MODEL_PATH)

# Simulate new incoming patient data

new_patients = pd.DataFrame([
{
"age": 75, "gender": "Male", "length_of_stay": 10, "num_prev_adm": 3,
"chronic_diabetes": 1, "chronic_hypertension": 1,
"lab_glucose": 145.0, "lab_creatinine": 1.4,
"discharge_type": "SkilledNursing", "followup_days": 7,
"smoker": 1, "rural_resident": 0
},
{
"age": 33, "gender": "Female", "length_of_stay": 2, "num_prev_adm": 0,
"chronic_diabetes": 0, "chronic_hypertension": 0,
"lab_glucose": 95.0, "lab_creatinine": 0.9,
"discharge_type": "Home", "followup_days": 14,
"smoker": 0, "rural_resident": 1
}
])

# Predict readmission probabilities

preds = model.predict(new_patients)
probs = model.predict_proba(new_patients)[:, 1]

# Combine predictions into a single dataframe

results = new_patients.copy()
results["Predicted_Readmit"] = preds
results["Readmit_Prob"] = probs.round(3)

# Save results for review

output_path = BASE_DIR / "data" / "processed" / "deployment_inference_sample.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output_path, index=False)

print("[✔] Deployment simulation complete.")
print(f"Results saved to {output_path}")
print(results)
