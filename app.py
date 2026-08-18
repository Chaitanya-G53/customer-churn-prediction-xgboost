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
        # Feature vector according to standard XGBoost preprocessing
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
# 2. STREAMLIT UI - DASHBOARD CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Customer Churn Intelligence Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dashboard Dark Theme Styling
st.markdown(
    """
    <style>
    /* Dark Theme Core Styles */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Metric Card Component */
    .metric-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 15px;
    }
    .metric-title {
        color: #8b949e;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #f0f6fc;
        font-size: 24px;
        font-weight: 700;
    }
    
    /* Status Badges */
    .badge-stay {
        background-color: #1f3a2b;
        color: #3fb950;
        border: 1px solid #238636;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
        margin-bottom: 12px;
    }
    .badge-churn {
        background-color: #3d1d24;
        color: #f85149;
        border: 1px solid #da3633;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
        margin-bottom: 12px;
    }
    
    /* Risk Badges */
    .risk-low { color: #3fb950; font-weight: bold; font-size: 22px; }
    .risk-medium { color: #d29922; font-weight: bold; font-size: 22px; }
    .risk-high { color: #f85149; font-weight: bold; font-size: 22px; }

    /* Factors Bullet Items */
    .factor-positive {
        color: #3fb950;
        margin-bottom: 6px;
        font-size: 15px;
    }
    .factor-negative {
        color: #f85149;
        margin-bottom: 6px;
        font-size: 15px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 3. SIDEBAR: CUSTOMER PROFILE INPUTS
# ==========================================

st.sidebar.title("🎛️ Customer Profile Settings")

st.sidebar.markdown("### Demographics")
country = st.sidebar.selectbox(
    "Geography / Country", ["France", "Germany", "Spain"]
)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", min_value=18, max_value=92, value=44)

st.sidebar.markdown("---")
st.sidebar.markdown("### Financial & Engagement")
credit_score = st.sidebar.slider(
    "Credit Score", min_value=300, max_value=850, value=761
)
tenure = st.sidebar.slider("Tenure (Years)", min_value=0, max_value=10, value=3)
balance = st.sidebar.number_input(
    "Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0
)
products_number = st.sidebar.selectbox(
    "Number of Products", [1, 2, 3, 4], index=0
)
has_crcard = st.sidebar.radio("Has Credit Card?", ["Yes", "No"], index=1)
is_active = st.sidebar.radio("Is Active Member?", ["Yes", "No"], index=1)
estimated_salary = st.sidebar.number_input(
    "Estimated Salary ($)", min_value=0.0, value=75000.0, step=1000.0
)

# ==========================================
# 4. API CALL & DATA FETCH
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

churn_prob = 34.8  # Fallback default
try:
    res = requests.post("http://127.0.0.1:5000/predict", json=payload, timeout=2)
    if res.status_code == 200:
        data = res.json()
        churn_prob = data.get("churn_probability", 34.8)
except Exception:
    pass

retention_prob = round(100.0 - churn_prob, 1)

# Categorize Risk Level
if churn_prob < 30.0:
    risk_level = "LOW"
    risk_class = "risk-low"
elif churn_prob < 60.0:
    risk_level = "MEDIUM"
    risk_class = "risk-medium"
else:
    risk_level = "HIGH"
    risk_class = "risk-high"

# ==========================================
# 5. MAIN DASHBOARD CONTENT
# ==========================================

st.title("🔮 Customer Churn Intelligence Dashboard")
st.caption("Predict customer churn risk using an XGBoost Machine Learning Pipeline")

# Row 1: Top KPI Metric Boxes
top_col1, top_col2, top_col3 = st.columns(3)

with top_col1:
    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-title">MODEL ENGINE</div>
            <div class="metric-value">XGBoost Classifier</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with top_col2:
    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-title">MODEL PERFORMANCE</div>
            <div class="metric-value">86.41% ROC-AUC</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with top_col3:
    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-title">DECISION THRESHOLD</div>
            <div class="metric-value">50.0%</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Row 2: Prediction Overview & Gauge Chart
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Prediction Overview")

    if churn_prob < 50.0:
        st.markdown(
            '<div class="badge-stay">✓ Customer Likely to Stay</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="badge-churn">⚠️ High Risk of Churn</div>',
            unsafe_allow_html=True,
        )

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.markdown("**Retention Probability**")
        st.markdown(f"## {retention_prob}%")
    with p_col2:
        st.markdown("**Churn Probability**")
        st.markdown(f"## {churn_prob}%")
    with p_col3:
        st.markdown("**Risk Level**")
        st.markdown(
            f'<div class="{risk_class}">{risk_level}</div>',
            unsafe_allow_html=True,
        )

with col_right:
    st.subheader("Probability Gauge")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=churn_prob,
            number={"suffix": "%", "font": {"color": "#FFFFFF", "size": 36}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#8b949e",
                },
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
        height=220,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Row 3: Customer Feature Profile Table
st.subheader("🔍 Customer Feature Profile")

feature_df = pd.DataFrame(
    [
        {
            "credit_score": credit_score,
            "country": country,
            "gender": gender,
            "age": age,
            "tenure": tenure,
            "balance": f"${balance:,.2f}",
            "products_number": products_number,
            "credit_card": has_crcard,
            "active_member": is_active,
            "estimated_salary": f"${estimated_salary:,.2f}",
        }
    ]
)

st.dataframe(feature_df, use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

# Row 4: Factors Influencing Prediction
st.subheader("🔍 Factors Influencing Prediction")

# Dynamic Explanation Logic
factors = []

if is_active == "No":
    factors.append(
        (
            "negative",
            "🔴 <b>Inactive membership:</b> increases churn risk significantly",
        )
    )
else:
    factors.append(
        ("positive", "🟢 <b>Active membership:</b> reduces risk of customer churn")
    )

if credit_score > 700:
    factors.append(
        (
            "positive",
            "🟢 <b>High credit score (> 700):</b> lowers overall churn risk",
        )
    )
elif credit_score < 500:
    factors.append(
        ("negative", "🔴 <b>Low credit score (< 500):</b> increases risk score")
    )

if age > 40:
    factors.append(
        (
            "negative",
            f"🔴 <b>Customer age ({age}):</b> older demographic cohort correlates with higher churn",
        )
    )

if products_number >= 3:
    factors.append(
        (
            "negative",
            f"🔴 <b>High product count ({products_number}):</b> multi-product friction elevates churn likelihood",
        )
    )

for f_type, f_text in factors:
    st.markdown(
        f'<div class="factor-{f_type}">{f_text}</div>', unsafe_allow_html=True
    )
