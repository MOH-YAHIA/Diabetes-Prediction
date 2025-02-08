import streamlit as st
import numpy as np
import joblib

# Load the trained model
rf_model = joblib.load(r"D:\track\diabetes\diabetes_model.joblib")

# Streamlit app title
st.title("Diabetes Prediction App for Females (Accuracy: 86%)")

# Sidebar title
st.sidebar.header("Enter Patient Details:")


# Input fields with feature descriptions
pregnancies = st.sidebar.number_input("Pregnancies (Number of times pregnant)", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose (Plasma glucose concentration in OGTT)", 0, 300, 120)
blood_pressure = st.sidebar.number_input("Blood Pressure (Diastolic BP in mm Hg)", 0, 200, 80)
skin_thickness = st.sidebar.number_input("Skin Thickness (Triceps skin fold thickness in mm)", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin (2-Hour serum insulin in mu U/ml)", 0, 900, 30)
bmi = st.sidebar.number_input("BMI (Body Mass Index: weight in kg/(height in m)^2)", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function (Genetic influence on diabetes)", 0.0, 3.0, 0.5)
age = st.sidebar.number_input("Age (Years)", 1, 120, 40)

# Make prediction when user clicks button
if st.sidebar.button("Predict"):
    # Prepare input data
    patient_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

    # Make prediction
    prediction = rf_model.predict(patient_data)
    
    # Show result
    if prediction[0] == 1:
        st.error("🚨 The model predicts **Diabetes Positive**!")
    else:
        st.success("✅ The model predicts **Diabetes Negative**!")
