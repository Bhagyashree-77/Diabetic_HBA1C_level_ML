import streamlit as st
import pandas as pd
import numpy as np
from backend.preprocessing import load_and_preprocess_data, get_fitted_preprocessor
from backend.preprocessing import  get_trained_model
from backend.models import run_classification_models

# Set metadata for better compatibility and SEO
st.set_page_config(page_title="Diabetes ML App", layout="wide")

# Global custom CSS
st.markdown("""
<style>
    * {
        -webkit-user-select: auto;
        user-select: auto;
    }
    label, input, select {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Page title
st.title("🩺 Diabetes Patient ML Prediction Suite")

# Sidebar options
task = st.sidebar.selectbox("Choose a Task", ["Classification", "Regression (Coming Soon)", "Clustering (Coming Soon)"], key="task_selector")

# Load data and preprocess once
@st.cache_data
def load_data():
    return load_and_preprocess_data("diabetic_data.csv")

X_train, X_test, y_train, y_test = load_data()

# Classification Section
if task == "Classification":
    st.header("📊 Predict Diabeties  Risk")

    model_name = st.selectbox(
        "Select a Classification Model:",
        ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"],
        key="model_selector"
    )

    if st.button("🚀 Train & Evaluate Model", key="train_button"):
        report, auc_score = run_classification_models(model_name, X_train, X_test, y_train, y_test)
        st.subheader("📈 Model Evaluation Report")
        st.text(report)
        st.metric("🔍 ROC AUC Score", f"{auc_score:.3f}")

elif task == "Regression (Coming Soon)":
    st.info("🚧 Regression module is under development. Stay tuned!")

elif task == "Clustering (Coming Soon)":
    st.info("🚧 Clustering module is under development. Stay tuned!")

# Patient prediction section
st.markdown("---")

# Form inputs
st.subheader("🧪 Predict HbA1c Level for a New Patient")

# Form input for features
with st.form("patient_form"):
    age = st.selectbox("Age Range", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
    time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 3)
    num_medications = st.slider("Number of Medications", 1, 50, 10)
    num_lab_procedures = st.slider("Number of Lab Procedures", 1, 132, 40)
    num_procedures = st.slider("Number of Procedures", 0, 6, 1)
    num_diagnoses = st.slider("Number of Diagnoses", 1, 16, 5)
    insulin = st.selectbox("Insulin Level", ["No", "Up", "Down", "Steady"])
    diabetesMed = st.selectbox("On Diabetes Medication?", ["Yes", "No"])
    submit = st.form_submit_button("Predict HbA1c Level")

# Sample row generator
@st.cache_data
def get_template_input():
    df = pd.read_csv("diabetic_data.csv")
    df = df.replace('?', np.nan)
    df = df.drop(columns=["encounter_id", "patient_nbr", "weight", "examide", "citoglipton"], errors='ignore')
    df = df[df['A1Cresult'].isin(['None', 'Norm', '>7', '>8'])].copy()
    df = df.drop(columns=['A1Cresult'])  # Drop original label

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].median())

    return df.iloc[0]

# Predict HbA1c for new data
if submit:
    template = get_template_input()
    template["age"] = age
    template["time_in_hospital"] = time_in_hospital
    template["num_medications"] = num_medications
    template["num_lab_procedures"] = num_lab_procedures
    template["num_procedures"] = num_procedures
    template["num_diagnoses"] = num_diagnoses
    template["insulin"] = insulin
    template["diabetesMed"] = diabetesMed

    new_data = pd.DataFrame([template])
    preprocessor = get_fitted_preprocessor()
    X_input = preprocessor.transform(new_data)

    model = get_trained_model(model_name, X_train, y_train)
    prediction = model.predict(X_input)[0]

    label_names = ['None', 'Norm', '>7', '>8']
    st.success(f"🔍 Predicted HbA1c Level: **{label_names[int(prediction)]}**")