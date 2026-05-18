import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = joblib.load("models/xgboost_model.pkl")
    return model

model = load_model()

st.title("❤️ Heart Disease Predictor")
st.markdown("Fill in the patient details below and click **Predict**.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=20, max_value=80, value=54)
    trestbps = st.number_input("Resting BP (mm Hg)", min_value=90, max_value=200, value=130)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=246)
    restecg = st.selectbox("Resting ECG", [0, 1, 2],
                format_func=lambda x: {0:"Normal", 1:"ST Abnormality", 2:"LV Hypertrophy"}[x])
    ca = st.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4])

with col2:
    sex = st.selectbox("Sex", [1, 0],
                format_func=lambda x: "Male" if x==1 else "Female")
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1],
                format_func=lambda x: "Yes" if x==1 else "No")
    thalach = st.number_input("Max Heart Rate", min_value=70, max_value=210, value=150)
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0,
                value=1.0, step=0.1)

with col3:
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                format_func=lambda x: {
                    0:"Typical Angina",
                    1:"Atypical Angina",
                    2:"Non-Anginal",
                    3:"Asymptomatic"}[x])
    exang = st.selectbox("Exercise Angina", [0, 1],
                format_func=lambda x: "Yes" if x==1 else "No")
    slope = st.selectbox("ST Slope", [0, 1, 2],
                format_func=lambda x: {0:"Upsloping", 1:"Flat", 2:"Downsloping"}[x])
    thal = st.selectbox("Thal", [0, 1, 2, 3],
                format_func=lambda x: {
                    0:"Normal",
                    1:"Fixed Defect",
                    2:"Reversible Defect",
                    3:"Unknown"}[x])

st.divider()

if st.button("🔍 Predict", type="primary", use_container_width=True):
    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak, slope, ca, thal
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("### Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease — Confidence: {probability*100:.1f}%")
    else:
        st.success(f"✅ Low Risk of Heart Disease — Confidence: {(1-probability)*100:.1f}%")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Prediction", "Heart Disease" if prediction==1 else "No Disease")
    col_b.metric("Probability", f"{probability*100:.1f}%")
    col_c.metric("Model", "XGBoost")

st.divider()
st.caption("Model Accuracy: 81.97% | ROC-AUC: 87.66% | Dataset: Heart Disease UCI (Kaggle)")