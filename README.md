# 🤖 AI Development Workflow: Patient Readmission Risk Prediction

An end-to-end **AI development pipeline** that predicts whether a patient is likely to be readmitted within **30 days of hospital discharge**.

This project demonstrates the complete AI lifecycle — from **data generation and preprocessing** to **model training, evaluation, and deployment simulation** — following modern MLOps and responsible AI practices.

---

## 🚀 Overview

Hospital readmissions drive up healthcare costs and strain limited medical resources.
By applying machine learning on clinical and demographic data, this model helps healthcare providers:

* Identify high-risk patients before discharge.
* Prioritize follow-up interventions.
* Reduce readmission rates and improve patient outcomes.

---

## 🧠 Workflow Architecture

```
Problem Definition → Data Collection → Preprocessing → Model Training
          ↓                 ↓                     ↓
   Evaluation ← Deployment Simulation ← Monitoring & Maintenance
```

Each stage is modularized within the `src/` directory for maintainability and reuse.

---

## ⚙️ Setup & Installation

**1. Clone the Repository**

```bash
git clone https://github.com/Manrutz/week_5_AI-SE_AI-Development-Workflow/tree/main
```

**2. Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate     # On macOS/Linux
venv\Scripts\activate        # On Windows
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## 🧩 Running the Pipeline

### 1. Data Preprocessing

Generates a synthetic hospital dataset and prepares it for modeling.

```bash
python src/data_preprocessing.py
```

### 2. Model Training

Trains a `RandomForestClassifier` using recall-optimized tuning for healthcare scenarios.

```bash
python src/model_training.py
```

### 3. Model Evaluation

Evaluates precision, recall, and accuracy; saves confusion matrix visualizations.

```bash
python src/evaluation.py
```

### 4. Deployment Simulation

Loads the trained pipeline and predicts readmission risk for new patients.

```bash
python src/deployment_simulation.py
```

---

## 📊 Model Performance Snapshot

| Dataset    | Accuracy | Precision | Recall |
| ---------- | -------- | --------- | ------ |
| Train      | 1.000    | 1.000     | 1.000  |
| Validation | 0.870    | 0.874     | 0.988  |
| Test       | 0.873    | 0.876     | 0.988  |

**Key Takeaway:**
The model achieves **high recall (≈ 0.99)** — ensuring nearly all at-risk patients are detected while maintaining strong precision.

---

## 🛡️ Ethical Considerations

* **Bias Monitoring:** The model should be routinely checked for demographic or socioeconomic bias.
* **Explainability:** Uses interpretable models (RandomForest/XGBoost) with feature importance visualization.
* **Privacy Compliance:** Synthetic data is used for development; real implementations must adhere to HIPAA or GDPR.

---

## 📂 Project Structure

```
AI_Development_Workflow/
│
├── data/
│   ├── raw/                # Synthetic generated data
│   └── processed/          # Train/val/test splits and inference samples
│
├── models/                 # Serialized trained models (.joblib)
│
├── src/                    # Modular Python scripts
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── deployment_simulation.py
│
├── workflow_pipeline.ipynb # End-to-end notebook version
├── README.md
└── requirements.txt
```

---

## 🧑🏽‍💻 Author

**Remmy Kipruto Tumo**
AI Software Engineer | Data Science Enthusiast
📫 [LinkedIn](https://www.linkedin.com/in/kipruto-tumo-a1630a374/)) · [GitHub](https://github.com/Manrutz/week_5_AI-SE_AI-Development-Workflow/new/main)

---

## 🪪 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.
