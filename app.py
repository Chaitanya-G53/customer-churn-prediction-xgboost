import os
import pickle
import threading
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request
import streamlit as st

# ==========================================
# 1. FLASK BACKEND SERVER
# ==========================================

flask_app = Flask(__name__)
MODEL_FILE = "Customer-Churn-Prediction-XGBOOST.pkl"

# Load trained XGBoost model
try:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None


@flask_app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": f"Model file '{MODEL_FILE}' not found."}), 500

    try:
        data = request.json
        # Feature order expected by the XGBoost model based on binary layout:
        # credit_score, age, tenure, balance, products_number, credit_card,
        # active_member, estimated_salary, country_France, country_Germany,
        # country_Spain, gender_Female, gender_Male
        features = [
            data.get("credit_score", 600),
            data.get("age", 40),
            data.get("tenure", 3),
            data.get("balance", 60000.0),
            data.get("products_number", 1),
            data.get("credit_card", 1),
            data.get("active_member", 1),
            data.get("estimated_salary", 50000.0),
            1 if data.get("country") == "France" else 0,
            1 if data.get("country") == "Germany" else 0,
            1 if data.get("country") == "Spain" else 0,
            1 if data.get("gender") == "Female" else 0,
            1 if data.get("gender") == "Male" else 0,
        ]

        features_array = np.array(features).reshape(1, -1)
        prediction = int(model.predict(features_array)[0])

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features_array)[0][1])
        else:
            probability = float(prediction)

        return jsonify(
            {
                "churn_prediction": prediction,
                "churn_probability": round(probability * 100, 2),
                "status": "Success",
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


def run_flask():
    flask_app.run(port=5000, debug=False, use_reloader=False)


# Start Flask server in background thread if not already running
if not any(thread.name == "FlaskThread" for thread in threading.enumerate()):
    flask_thread = threading.Thread(target=run_flask, name="FlaskThread", daemon=True)
    flask_thread.start()


# ==========================================
# 2. STREAMLIT UI (DARK THEME & IN-CONTRAST TEXT)
# ==========================================

st.set_page_config(
    page_title="Customer Churn Predictor", page_icon="📉", layout="wide"
)

# Custom High-Contrast Dark Theme Styling
st.markdown(
    """
    <style>
    /* Dark background for app */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Headers & Text Formatting */
    h1, h2, h3, h4, h5, h6, label, .stMarkdown {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    
    /* Input Boxes High Contrast Styling */
    input, select, div[data-baseweb="select"] {
        background-color: #1F2937 !important;
        color: #00F0FF !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }

    /* Prediction Card Display */
    .metric-card {
        background-color: #161B22;
        border: 2px solid #30363D;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
    }
    .churn-high {
        color: #FF4B4B;
        font-size: 28px;
        font-weight: bold;
    }
    .churn-low {
        color: #00E676;
        font-size: 28px;
        font-weight: bold;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("📉 Customer Churn Prediction Dashboard")
st.markdown(
    "Fill in the customer attributes below to evaluate the likelihood of churn using the XGBoost backend service."
)
st.markdown("---")

# Input Form organized into columns for better usability
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographic Factors")
    country = st.selectbox("Country / Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=38, step=1)

with col2:
    st.subheader("💳 Financial Factors")
    credit_score = st.number_input(
        "Credit Score", min_value=300, max_value=850, value=650, step=1
    )
    balance = st.number_input(
        "Account Balance ($)", min_value=0.0, value=75000.0, step=1000.0
    )
    estimated_salary = st.number_input(
        "Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0
    )

with col3:
    st.subheader("🏦 Engagement Factors")
    tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5)
    products_number = st.slider(
        "Number of Products", min_value=1, max_value=4, value=2
    )
    has_crcard = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])

st.markdown("---")

# Predict Button and Request to Flask Service
if st.button("🚀 Analyze Churn Risk", use_container_width=True):
    payload = {
        "credit_score": credit_score,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "products_number": products_number,
        "credit_card": 1 if has_crcard == "Yes" else 0,
        "active_member": 1 if is_active == "Yes" else 0,
        "estimated_salary": estimated_salary,
        "country": country,
        "gender": gender,
    }

    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            prediction = result["churn_prediction"]
            probability = result["churn_probability"]

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-color: #FF4B4B;">
                            <h4>Churn Risk Status</h4>
                            <p class="churn-high">⚠️ HIGH RISK (WILL CHURN)</p>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-color: #00E676;">
                            <h4>Churn Risk Status</h4>
                            <p class="churn-low">✅ LOW RISK (RETAINED)</p>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

            with res_col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4>Churn Probability</h4>
                        <p style="font-size: 28px; font-weight: bold; color: #00F0FF;">{probability}%</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

        else:
            st.error(f"Backend API Error: {response.json().get('error')}")

    except Exception as e:
        st.error(
            f"Failed to connect to the Flask API. Make sure local port 5000 is open. Details: {e}"
        )
