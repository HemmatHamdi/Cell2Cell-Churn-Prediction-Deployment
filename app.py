import streamlit as st
import joblib
import pandas as pd
import os

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Cell2Cell Churn Prediction",
    page_icon="📱",
    layout="centered"
)

# ==========================================
# Load Model
# ==========================================

MODEL_PATH = os.path.join("model", "xgboost_model.pkl")
FEATURES_PATH = os.path.join("model", "selected_features.pkl")

try:
    model = joblib.load(MODEL_PATH)
    selected_features = joblib.load(FEATURES_PATH)

except Exception as e:
    st.error("Model Loading Failed")
    st.exception(e)
    st.stop()

# ==========================================
# Features
# ==========================================

input_features = [
    "MonthlyRevenue",
    "MonthlyMinutes",
    "TotalRecurringCharge",
    "OverageMinutes",
    "DroppedCalls",
    "CustomerCareCalls",
    "CallForwardingCalls"
]

# ==========================================
# App Interface
# ==========================================

st.title("📱 Cell2Cell Customer Churn Prediction")

st.write(
    "Enter customer details to predict the probability of customer churn."
)

st.divider()

# Store user inputs
input_data = {}

for feature in input_features:
    input_data[feature] = st.number_input(
        label=feature,
        min_value=0.0,
        value=0.0
    )


# ==========================================
# Prediction
# ==========================================

if st.button("Predict Churn"):

    # Create dataframe with all model features
    model_input = {feature: 0 for feature in selected_features}

    # Add entered values
    for feature in input_features:
        model_input[feature] = input_data[feature]

    input_df = pd.DataFrame([model_input])

    # Ensure same feature order as training
    input_df = input_df[selected_features]

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:
        st.error("⚠️ Customer Will Churn")
    else:
        st.success("✅ Customer Will Stay")

    st.metric(
        label="Churn Probability",
        value=f"{probability * 100:.2f}%"
    )