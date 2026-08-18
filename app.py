import os
import pickle
import threading
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
except Exception:
    model = None


@flask_app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": f"Model file '{MODEL_FILE}' not found."}), 500

    try:
        data = request.json

        # Raw features strictly aligned with model inputs
        # Expected order: CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Country, Gender
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
                "churn_probability": round(probability * 100, 1),
                "status": "Success",
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


def run_flask():
    flask_app.run(port=5000, debug=False, use_reloader=False)


if not any(thread.name == "FlaskThread" for thread in threading.enumerate()):
    flask_thread = threading.Thread(target=run_flask, name="FlaskThread", daemon=True)
    flask_thread.start()


# ==========================================
# 2. STREAMLIT UI CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark Theme Polish
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0e14;
        color: #e6edf3;
    }
    section[data-testid="stSidebar"] {
        background-color: #151b23;
        border-right: 1px solid #30363d;
    }
    .metric-box {
        background: #151b23;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .badge-stay {
        background-color: #1f3a2b;
        color: #3fb950;
        border: 1px solid #238636;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-churn {
        background-color: #3d1d24;
        color: #f85149;
        border: 1px solid #da3633;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 3. SIDEBAR: CUSTOMER ATTRIBUTES
# ==========================================

st.sidebar.title("🎛️ Customer Profile")

st.sidebar.subheader("Demographics")
country = st.sidebar.selectbox("Country", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 92, 44)

st.sidebar.subheader("Financial & Activity")
credit_score = st.sidebar.slider("Credit Score", 300, 850, 761)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3)
balance = st.sidebar.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
products_number = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4], index=0)
has_crcard = st.sidebar.radio("Has Credit Card?", ["Yes", "No"], index=1)
is_active = st.sidebar.radio("Is Active Member?", ["Yes", "No"], index=1)
estimated_salary = st.sidebar.number_input("Estimated Salary ($)", min_value=0.0, value=75000.0, step=1000.0)

# ==========================================
# 4. PREDICTION INFERENCE
# ==========================================

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

churn_prob = 0.0
try:
    res = requests.post("http://127.0.0.1:5000/predict", json=payload, timeout=2)
    if res.status_code == 200:
        churn_prob = res.json().get("churn_probability", 0.0)
except Exception:
    pass

retention_prob = round(100.0 - churn_prob, 1)

# Dynamic Risk Determination
if churn_prob < 30.0:
    risk_level, risk_color = "LOW", "#3fb950"
elif churn_prob < 60.0:
    risk_level, risk_color = "MEDIUM", "#d29922"
else:
    risk_level, risk_color = "HIGH", "#f85149"

# ==========================================
# 5. DASHBOARD LAYOUT
# ==========================================

st.title("🔮 Customer Churn Intelligence Dashboard")
st.caption("Real-time inference using your loaded XGBoost model pipeline")

st.markdown("<br>", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Prediction Overview")

    if churn_prob < 50.0:
        st.markdown('<div class="badge-stay">✓ Customer Likely to Stay</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-churn">⚠️ High Risk of Churn</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown("**Retention Prob.**")
        st.markdown(f"### {retention_prob}%")
    with m2:
        st.markdown("**Churn Prob.**")
        st.markdown(f"### {churn_prob}%")
    with m3:
        st.markdown("**Risk Level**")
        st.markdown(f"<h3 style='color: {risk_color};'>{risk_level}</h3>", unsafe_allow_html=True)

with col_right:
    st.subheader("Probability Gauge")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=churn_prob,
            number={"suffix": "%", "font": {"color": "#FFFFFF", "size": 32}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                "bar": {"color": "#FFFFFF", "thickness": 0.15},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#2ea44f"},
                    {"range": [30, 60], "color": "#d29922"},
                    {"range": [60, 100], "color": "#cb2431"},
                ],
            },
        )
    )

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Feature Summary Table
st.subheader("🔎 Customer Profile Summary")

feature_df = pd.DataFrame(
    [
        {
            "Credit Score": credit_score,
            "Country": country,
            "Gender": gender,
            "Age": age,
            "Tenure": f"{tenure} yrs",
            "Balance": f"${balance:,.2f}",
            "Products": products_number,
            "Credit Card": has_crcard,
            "Active Member": is_active,
            "Salary": f"${estimated_salary:,.2f}",
        }
    ]
)

st.dataframe(feature_df, use_container_width=True, hide_index=True)

# Key Risk Drivers
st.subheader("💡 Key Risk Factors")

factors = []
if is_active == "No":
    factors.append("🔴 **Inactive membership:** Significantly increases likelihood of churn.")
else:
    factors.append("🟢 **Active member:** Lowers churn risk.")

if age > 40:
    factors.append(f"🔴 **Age factor ({age}):** Customers over 40 show higher historical churn rates.")

if products_number >= 3:
    factors.append(f"🔴 **Product count ({products_number}):** Having 3+ products correlates with elevated friction.")

for factor in factors:
    st.markdown(factor)
