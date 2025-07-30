import numpy as np
import streamlit as st
import joblib

# First load the trained model
model = joblib.load('churn_model.pkl')

# Title for the app
st.title("Customer Churn Prediction App")

st.header("Enter Customer Information:")

gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (in months)", 0, 72, 1)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Map input data to the model's expected format
input_data = np.array([
    1 if gender == "Male" else 0,
    senior_citizen,
    1 if partner == "Yes" else 0,
    1 if dependents == "Yes" else 0,
    tenure,
    monthly_charges,
    total_charges,
    0 if contract == "Month-to-month" else 1 if contract == "One year" else 2,
    0 if internet_service == "DSL" else 1 if internet_service == "Fiber optic" else 2
]).reshape(1, -1)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is likely to stay.")
