"""
app.py — Streamlit web app for heart-disease-xgboost
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Heart Disease Xgboost",
    page_icon="🔬",
    layout="centered",
)

@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_model.pkl")

model = load_model()

# ── UI ────────────────────────────────────────────────────────────────
st.title("🔬 Heart Disease Xgboost")
st.markdown("Enter the patient details below and click **Predict** to see the result.")

st.divider()

# ── Input fields (customize per your features) ───────────────────────
col1, col2 = st.columns(2)

with col1:
    age  = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex  = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    cp   = st.selectbox("Chest Pain Type", [0, 1, 2, 3])

with col2:
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol     = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs      = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

# Add more inputs as needed ↑

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    # Build input array — match your training feature order
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs]])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("### Result")
    if prediction == 1:
        st.error(f"High Risk  —  Confidence: {probability*100:.1f}%")
    else:
        st.success(f"Low Risk  —  Confidence: {(1-probability)*100:.1f}%")

st.divider()
st.caption("Built with XGBoost · Streamlit · by [Your Name]")
