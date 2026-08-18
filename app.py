import os
import pickle
import numpy as np
from flask import Flask, jsonify, request
from pydantic import BaseModel, Field, ValidationError

app = Flask(__name__)

# Model path config
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")

# Load XGBoost Classifier Model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"Warning: Could not load model from {MODEL_PATH}. Error: {e}")


# Input Schema Validation matching the 13 model features
class CustomerData(BaseModel):
    credit_score: int = Field(..., example=600)
    age: int = Field(..., example=40)
    tenure: int = Field(..., example=3)
    balance: float = Field(..., example=60000.0)
    products_number: int = Field(..., example=2)
    credit_card: int = Field(..., example=1)
    active_member: int = Field(..., example=1)
    estimated_salary: float = Field(..., example=50000.0)
    country_France: int = Field(..., example=1)
    country_Germany: int = Field(..., example=0)
    country_Spain: int = Field(..., example=0)
    gender_Female: int = Field(..., example=0)
    gender_Male: int = Field(..., example=1)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    json_data = request.get_json(silent=True)
    if not json_data:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

    try:
        # Validate JSON payload against expected fields
        validated_data = CustomerData(**json_data)
    except ValidationError as err:
        return jsonify({"error": "Validation error", "details": err.errors()}), 422

    # Map inputs to the exact feature order expected by the model
    features = [
        validated_data.credit_score,
        validated_data.age,
        validated_data.tenure,
        validated_data.balance,
        validated_data.products_number,
        validated_data.credit_card,
        validated_data.active_member,
        validated_data.estimated_salary,
        validated_data.country_France,
        validated_data.country_Germany,
        validated_data.country_Spain,
        validated_data.gender_Female,
        validated_data.gender_Male,
    ]

    input_array = np.array([features])

    # Run inference
    probability = float(model.predict_proba(input_array)[0][1])
    prediction = int(model.predict(input_array)[0])

    return (
        jsonify(
            {
                "prediction": prediction,
                "probability": round(probability, 4),
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
