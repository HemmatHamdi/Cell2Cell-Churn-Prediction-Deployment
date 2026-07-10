from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ==========================================
# Load Model
# ==========================================

MODEL_PATH = os.path.join("model", "xgboost_model.pkl")
FEATURES_PATH = os.path.join("model", "selected_features.pkl")

model = joblib.load(MODEL_PATH)
selected_features = joblib.load(FEATURES_PATH)

# ==========================================
# Features shown in the website
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
# Home Page
# ==========================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        features=input_features
    )

# ==========================================
# Prediction
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # Create all model features with default value = 0
    input_data = {feature: 0 for feature in selected_features}

    # Replace only the entered features
    for feature in input_features:

        value = request.form.get(feature)

        if value == "" or value is None:
            value = 0

        input_data[feature] = float(value)

    # Arrange columns exactly as the model expects
    input_df = pd.DataFrame([input_data])
    input_df = input_df[selected_features]

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        result = "⚠️ Customer Will Churn"
    else:
        result = "✅ Customer Will Stay"

    return render_template(
        "result.html",
        prediction=result,
        probability=round(probability * 100, 2)
    )

# ==========================================
# Run App
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)