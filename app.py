"""
Diabetes Prediction Using ML - Flask Web Application
This script serves the web interface, validates patient inputs,
loads the trained Decision Tree model, and renders prediction results.
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

# Initialize Flask application
app = Flask(__name__)

# Expected input features in exact training order
FEATURE_KEYS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Path to trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.pkl")

# Load model once at application startup
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Loaded trained Decision Tree model from {MODEL_PATH}")
    except Exception as e:
        print(f"Warning: Failed to load model file: {e}")
else:
    print(f"Warning: Model file not found at {MODEL_PATH}. Run train_model.py first.")


def validate_and_parse_inputs(form_data):
    """
    Validates form data and converts inputs to appropriate numeric types.
    Returns: (parsed_feature_list, error_message)
    """
    parsed_values = []
    
    # 1. Check all required fields are present and non-empty
    for key in FEATURE_KEYS:
        val_str = form_data.get(key, "").strip()
        if not val_str:
            return None, f"Please fill in all required fields. Missing: {key}"
        
        # 2. Check value is numeric
        try:
            val_num = float(val_str)
        except ValueError:
            return None, f"Invalid value for '{key}'. Please enter a valid number."
        
        # 3. Domain validations
        if key == "Pregnancies":
            if val_num < 0 or not val_num.is_integer():
                return None, "Number of pregnancies must be a non-negative whole integer (0 or more)."
        elif key == "Glucose":
            if val_num <= 0 or val_num > 500:
                return None, "Glucose level must be a realistic positive value (typically 40 to 400 mg/dL)."
        elif key == "BloodPressure":
            if val_num < 0 or val_num > 300:
                return None, "Blood Pressure must be a non-negative realistic value (mm Hg)."
        elif key == "SkinThickness":
            if val_num < 0 or val_num > 120:
                return None, "Skin Thickness must be a non-negative realistic value (mm)."
        elif key == "Insulin":
            if val_num < 0 or val_num > 1000:
                return None, "Insulin level must be a non-negative realistic value (mu U/ml)."
        elif key == "BMI":
            if val_num <= 5.0 or val_num > 90.0:
                return None, "BMI must be a reasonable positive number (typically 10.0 to 70.0)."
        elif key == "DiabetesPedigreeFunction":
            if val_num < 0 or val_num > 5.0:
                return None, "Diabetes Pedigree Function must be a non-negative number (typically 0.05 to 2.5)."
        elif key == "Age":
            if val_num <= 0 or val_num > 120 or not val_num.is_integer():
                return None, "Age must be a positive whole integer between 1 and 120."
        
        parsed_values.append(val_num)

    return parsed_values, None


@app.route("/", methods=["GET"])
def index():
    """
    Renders the homepage containing the patient data entry form,
    model explanation, and academic project information.
    """
    return render_template("index.html", form_values={})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives submitted health parameters, validates values,
    runs the Decision Tree model, and renders the result page.
    """
    global model
    # Retain submitted form values to preserve them in case of validation error
    form_values = request.form.to_dict()

    # Ensure model is available
    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
            except Exception:
                pass
        if model is None:
            return render_template(
                "index.html",
                error="The trained machine learning model (diabetes_model.pkl) was not found. Please run train_model.py first.",
                form_values=form_values
            )

    try:
        # Validate inputs
        features, error = validate_and_parse_inputs(request.form)
        if error:
            return render_template("index.html", error=error, form_values=form_values)

        # Generate prediction using the loaded Decision Tree Classifier
        input_df = pd.DataFrame([features], columns=FEATURE_KEYS)
        prediction = int(model.predict(input_df)[0])

        # Prepare submitted patient data dictionary for the result card
        patient_data = {
            "Pregnancies": int(features[0]),
            "Glucose": features[1],
            "Blood Pressure": features[2],
            "Skin Thickness": features[3],
            "Insulin": features[4],
            "BMI": features[5],
            "Diabetes Pedigree Function": features[6],
            "Age": int(features[7])
        }

        # Render result template with classification output
        return render_template(
            "result.html",
            prediction=prediction,
            patient_data=patient_data
        )

    except Exception:
        # Catch unexpected errors to prevent exposing Python traces to users
        return render_template(
            "index.html",
            error="An unexpected error occurred while processing the prediction. Please verify your inputs and try again.",
            form_values=form_values
        )


if __name__ == "__main__":
    # Run development server on port 5000
    app.run(host="127.0.0.1", port=5000, debug=True)
