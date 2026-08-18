import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration & Dark Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Canvas Background */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }

    /* Headers & Typography */
    h1, h2, h3, h4 {
        color: #F8FAFC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 700;
    }
    
    /* Dark Theme Card Containers */
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    /* Status Badges */
    .status-badge-stay {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(74, 222, 128, 0.3);
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .status-badge-churn {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }

    /* Risk Level Indicators */
    .risk-low { color: #4ADE80; font-weight: 700; }
    .risk-med { color: #FACC15; font-weight: 700; }
    .risk-high { color: #F87171; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "customer_churn_xgboost_pipeline.pkl"

# -----------------------------------------------------------------------------
# 2. Model Loader & Feature Mapping
# -----------------------------------------------------------------------------
@st.cache_resource
def load_churn_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model file (`{path}`) not found. Ensure the pipeline file is saved in the working directory.")
        st.stop()
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load the model pipeline: {str(e)}")
        st.stop()

model = load_churn_model(MODEL_PATH)

# Baseline statistical averages for comparative directional risk analysis
BASELINE_MEANS = {
    "credit_score": 650.0,
    "age": 38.0,
    "tenure": 5.0,
    "balance": 76000.0,
    "products_number": 1.5,
    "estimated_salary": 100000.0
}

# -----------------------------------------------------------------------------
# 3. Sidebar Profile Controls
# -----------------------------------------------------------------------------
st.sidebar.header("📋 Customer Profile Settings")

with st.sidebar:
    st.subheader("Demographics")
    country = st.selectbox("Geography / Country", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 100, 38)
    
    st.subheader("Financial & Engagement")
    credit_score = st.slider("Credit Score", 300, 850, 650)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
    products_number = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
    credit_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
    active_member = st.radio("Is Active Member?", ["Yes", "No"], horizontal=True)
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=75000.0, step=1000.0)

# Canonical schema alignment
input_df = pd.DataFrame([{
    "credit_score": credit_score,
    "country": country,
    "gender": gender,
    "age": age,
    "tenure": tenure,
    "balance": balance,
    "products_number": products_number,
    "credit_card": 1 if credit_card == "Yes" else 0,
    "active_member": 1 if active_member == "Yes" else 0,
    "estimated_salary": estimated_salary
}])

# -----------------------------------------------------------------------------
# 4. Prediction Execution
# -----------------------------------------------------------------------------
try:
    probabilities = model.predict_proba(input_df)[0]
    stay_prob = probabilities[0]
    churn_prob = probabilities[1]
    prediction = int(churn_prob >= 0.5)
except Exception as err:
    st.error(f"Error executing prediction pipeline: {err}")
    st.stop()

# Risk classification tier
if churn_prob < 0.30:
    risk_level = "LOW"
    risk_class = "risk-low"
elif churn_prob < 0.60:
    risk_level = "MEDIUM"
    risk_class = "risk-med"
else:
    risk_level = "HIGH"
    risk_class = "risk-high"

# -----------------------------------------------------------------------------
# 5. Dashboard Layout
# -----------------------------------------------------------------------------
st.title("🔮 Customer Churn Intelligence Dashboard")
st.caption("Predict customer churn risk using an XGBoost Machine Learning Pipeline")

# Top Metrics Row
mcol1, mcol2, mcol3 = st.columns(3)
with mcol1:
    st.markdown("""
    <div class="metric-card">
        <span style="color:#94A3B8; font-size:0.85rem;">MODEL ENGINE</span>
        <h3 style="margin:4px 0 0 0;">XGBoost Classifier</h3>
    </div>
    """, unsafe_allow_html=True)

with mcol2:
    st.markdown("""
    <div class="metric-card">
        <span style="color:#94A3B8; font-size:0.85rem;">MODEL PERFORMANCE</span>
        <h3 style="margin:4px 0 0 0; color:#38BDF8;">86.41% ROC-AUC</h3>
    </div>
    """, unsafe_allow_html=True)

with mcol3:
    st.markdown("""
    <div class="metric-card">
        <span style="color:#94A3B8; font-size:0.85rem;">DECISION THRESHOLD</span>
        <h3 style="margin:4px 0 0 0;">50.0%</h3>
    </div>
    """, unsafe_allow_html=True)

# Main Result & Analytics Cards
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Prediction Overview")
    
    if prediction == 1:
        st.markdown('<div class="status-badge-churn">⚠️ Likely to Churn</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge-stay">✅ Customer Likely to Stay</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Retention Probability", f"{stay_prob * 100:.1f}%")
    res_col2.metric("Churn Probability", f"{churn_prob * 100:.1f}%")
    
    with res_col3:
        st.markdown("**Risk Level**")
        st.markdown(f'<span class="{risk_class}" style="font-size:1.4rem;">{risk_level}</span>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Probability Gauge")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob * 100,
        number={'suffix': "%", 'font': {'size': 32, 'color': '#F8FAFC'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#EF4444" if churn_prob >= 0.5 else "#22C55E"},
            'bgcolor': "#0F172A",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.2)'},
                {'range': [30, 60], 'color': 'rgba(234, 179, 8, 0.2)'},
                {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#F8FAFC", 'width': 3},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Feature Profile Table
# -----------------------------------------------------------------------------
st.subheader("🔍 Customer Feature Profile")
st.dataframe(
    input_df.style.format({
        "balance": "${:,.2f}",
        "estimated_salary": "${:,.2f}",
        "credit_score": "{:d}",
        "age": "{:d}",
        "tenure": "{:d}",
        "products_number": "{:d}",
        "credit_card": lambda x: "Yes" if x == 1 else "No",
        "active_member": lambda x: "Yes" if x == 1 else "No"
    }),
    use_container_width=True
)

# -----------------------------------------------------------------------------
# 7. Dynamic Directional Risk Factors
# -----------------------------------------------------------------------------
st.subheader("🔎 Factors Influencing Prediction")

influences = []

# Active Membership Assessment
if active_member == "Yes":
    influences.append("🟢 **Active membership**: lowers churn risk")
else:
    influences.append("🔴 **Inactive membership**: increases churn risk")

# Age Deviation Assessment
if age > 45:
    influences.append("🔴 **Age above baseline average (38)**: increases churn risk")
elif age < 30:
    influences.append("🟢 **Younger demographic segment**: lowers churn risk")

# Product Portfolio Volume
if products_number >= 3:
    influences.append("🔴 **High product count (3+)**: increases churn risk")
elif products_number == 2:
    influences.append("🟢 **Optimal product engagement (2 products)**: lowers churn risk")

# Balance Deviation
if balance > 100000:
    influences.append("🔴 **High account balance**: elevates sensitivity to churn")

# Credit Score Impact
if credit_score < 500:
    influences.append("🔴 **Low credit score (< 500)**: increases financial instability risk")
elif credit_score > 700:
    influences.append("🟢 **High credit score (> 700)**: lowers overall churn risk")

for infl in influences:
    st.markdown(infl)
