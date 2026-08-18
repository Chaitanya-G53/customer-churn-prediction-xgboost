import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Dark Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme Custom CSS
st.markdown("""
<style>
    /* Main Canvas Background */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Sidebar Background & Text */
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
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    /* Status Badges */
    .status-badge-stay {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(74, 222, 128, 0.3);
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    
    .status-badge-churn {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    
    /* Input Control Overrides */
    div[data-baseweb="input"] {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        border-color: #334155 !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        border-color: #334155 !important;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "customer_churn_xgboost_pipeline.pkl"

# -----------------------------------------------------------------------------
# Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def load_churn_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model file standard specified (`{path}`) not found. Ensure the pipeline file is saved in the working directory.")
        st.stop()
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load the model pipeline: {str(e)}")
        st.stop()

model = load_churn_model(MODEL_PATH)

# -----------------------------------------------------------------------------
# UI Header & Sidebar Controls
# -----------------------------------------------------------------------------
st.title("🔮 Customer Churn Intelligence Dashboard")
st.caption("Predict customer retention risk in real-time using XGBoost ML Pipeline")

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

# -----------------------------------------------------------------------------
# Data Processing & Prediction
# -----------------------------------------------------------------------------
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

try:
    probabilities = model.predict_proba(input_df)[0]
    stay_prob = probabilities[0]
    churn_prob = probabilities[1]
    prediction = int(churn_prob >= 0.5)
except Exception as err:
    st.error(f"Error making prediction: {err}")
    st.stop()

# -----------------------------------------------------------------------------
# Main Visualizations & Insights
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.markdown('<div class="status-badge-churn">⚠️ High Risk of Churn</div>', unsafe_allow_html=True)
        st.markdown(f"### Probability of Churn: <span style='color:#F87171;'>{churn_prob * 100:.1f}%</span>", unsafe_allow_html=True)
        st.write("This customer exhibits characteristics typically associated with account closure.")
    else:
        st.markdown('<div class="status-badge-stay">✅ Customer Likely to Stay</div>', unsafe_allow_html=True)
        st.markdown(f"### Retention Probability: <span style='color:#4ADE80;'>{stay_prob * 100:.1f}%</span>", unsafe_allow_html=True)
        st.write("This customer exhibits healthy engagement metrics.")
        
    st.markdown("---")
    st.markdown("**Key Risk Factors Identified:**")
    risk_factors = []
    if age > 45:
        risk_factors.append("• Older demographic segment (Age > 45)")
    if products_number >= 3:
        risk_factors.append("• High product count (> 2 products often indicates instability)")
    if active_member == "No":
        risk_factors.append("• Inactive membership status")
    if country == "Germany":
        risk_factors.append("• Geographical higher risk region (Germany)")
        
    if risk_factors:
        for factor in risk_factors:
            st.write(factor)
    else:
        st.write("• No major elevated risk flags detected.")
        
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Probability Distribution")
    
    # Dark Mode Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob * 100,
        number={'suffix': "%", 'font': {'size': 36, 'color': '#F8FAFC'}},
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
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Feature Summary Table
st.subheader("🔍 Evaluated Feature Profile")
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
