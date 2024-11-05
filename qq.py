import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('ET.pkl')

# Define the names of the prediction variables
feature_names = [
    "uWBC", "Double-J_stent_duration", "WBC", "ALT", "CR",
    "Albumin", "Stone_burden", "Surgical_Duration"
]



# Set up the Streamlit web interface
st.title("Predictor of Sepsis Risk")

# Collect user input
uWBC = st.number_input("Urinary WBC (uWBC):", min_value=0, max_value=500, value=100)
double_j_duration = st.number_input("Double-J stent duration (days):", min_value=0, max_value=365, value=30)
WBC = st.number_input("White Blood Cell Count (WBC):", min_value=0, max_value=50, value=10)
ALT = st.number_input("Alanine transaminase (ALT):", min_value=0, max_value=500, value=35)
CR = st.number_input("Creatinine (CR):", min_value=0, max_value=10, value=1)
Albumin = st.number_input("Albumin (g/dL):", min_value=1, max_value=5, value=4)
stone_burden = st.number_input("Stone burden (mm^2):", min_value=0, max_value=1000, value=50)
surgical_duration = st.number_input("Surgical Duration (minutes):", min_value=0, max_value=600, value=90)

# Convert the input features to an array for model processing
feature_values = [uWBC, double_j_duration, WBC, ALT, CR, Albumin, stone_burden, surgical_duration]
features = np.array([feature_values])

# Make predictions when the user clicks "Predict"
if st.button("Predict"):
    # Predict the class (sepsis or no sepsis)
    predicted_class = model.predict(features)[0]

    # Predict the probabilities
    predicted_proba = model.predict_proba(features)[0]

    # Display the prediction results
    st.write(f"**Predicted Class:** {'Sepsis' if predicted_class == 1 else 'No Sepsis'}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Provide advice based on the prediction
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of sepsis. "
            f"The model predicts that your probability of having sepsis is {probability:.1f}%. "
            "Please consult your doctor for further evaluation and potential treatments."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of sepsis. "
            f"The model predicts that your probability of not having sepsis is {probability:.1f}%. "
            "Keep monitoring your health and consult a doctor if you have any concerns."
        )
    st.write(advice)


