# 🩺 Diabetic HbA1c Level Classification using Machine Learning

This project predicts and classifies **HbA1c levels** in diabetic patients using machine learning models. It aims to provide healthcare insights to aid in early intervention, risk stratification, and decision-making.

## 📌 Project Goals

- ✅ Preprocess and clean diabetes patient data.
- ✅ Train ML models (Logistic Regression, Random Forest) to classify HbA1c levels.
- ✅ Build a full-stack **Streamlit web application** for predictions and model evaluation.
- ✅ Improve healthcare delivery in support of **UN SDG 9: Industry, Innovation & Infrastructure**.

---

## 🧠 Machine Learning Models

We trained and evaluated multiple models using the `diabetic_data.csv` dataset:
- **Logistic Regression**
- **Random Forest Classifier**

Target: HbA1c result (`>7`, `<=7`, `None`)  
Features: Patient demographics, medications, diagnoses, hospital visits, and more.

---

## ⚙️ Project Structure

```bash
Diabetic_HBA1C_level_ML/
├── app.py                         # Streamlit frontend
├── diabetic_data.csv              # Original dataset (not uploaded due to size)
├── backend/
│   ├── __init__.py
│   ├── preprocessing.py           # Data clean
---

## 🧠 Models Used

- **Logistic Regression**
- **Random Forest Classifier**

Both models were trained to classify patients' HbA1c levels based on features like:
- Demographics
- Medical specialties
- Hospital visits
- Prescriptions
- Diagnoses and lab results

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Bhagyashree-77/Diabetic_HBA1C_level_ML.git
cd Diabetic_HBA1C_level_ML
ing and transformation
│   ├── models.py                  # ML model training and loading
├── logistic_model.pkl             # Trained Logistic Regression model
├── preprocessor_pipeline.pkl      # Preprocessing pipeline (Pickle)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Ignore large files
└── README.md                      # This file

python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

pip install -r requirements.txt
🧪 Features
Interactive UI for uploading data and generating predictions.

Switch between ML models dynamically.

Display model accuracy and confusion matrix.

Automated preprocessing and prediction pipeline.

Clean visualizations and intuitive layout.

