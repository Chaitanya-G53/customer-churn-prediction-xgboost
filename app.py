import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)

# Load Trained Model Pipeline
@st.cache_resource
def load_model():
    try:
        model = joblib.load("Customer_churn_xgboost_pipeline.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

pipeline = load_model()

# Header
st.markdown('<div class="main-header">📊 Customer Churn Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Input customer demographic and account metrics to evaluate retention risk.</div>', unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("📋 Customer Profile")

country = st.sidebar.selectbox("Country / Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
age = st.sidebar.slider("Age", min_value=18, max_value=100, value=38)
tenure = st.sidebar.slider("Tenure (Years)", min_value=0, max_value=10, value=5)

st.sidebar.divider()
st.sidebar.header("💳 Account Details")

credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
balance = st.sidebar.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
estimated_salary = st.sidebar.number_input("Estimated Salary ($)", min_value=0.0, value=100000.0, step=1000.0)
products_number = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4], index=0)

credit_card = st.sidebar.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
active_member = st.sidebar.radio("Is Active Member?", ["Yes", "No"], horizontal=True)

# Map binary features to numerical format expected by pipeline
has_cr_card = 1 if credit_card == "Yes" else 0
is_active = 1 if active_member == "Yes" else 0

# Create input DataFrame matching pipeline feature names
input_data = pd.DataFrame([{
    "credit_score": credit_score,
    "country": country,
    "gender": gender,
    "age": age,
    "tenure": tenure,
    "balance": balance,
    "products_number": products_number,
    "credit_card": has_cr_card,
    "active_member": is_active,
    "estimated_salary": estimated_salary
}])

# Layout Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("👤 Selected Attributes")
    
    # Display summary table of input data
    display_df = pd.DataFrame({
        "Attribute": ["Country", "Gender", "Age", "Credit Score", "Tenure", "Balance", "Products", "Credit Card", "Active Member", "Salary"],
        "Value": [country, gender, f"{age} yrs", credit_score, f"{tenure} yrs", f"${balance:,.2f}", products_number, credit_card, active_member, f"${estimated_salary:,.2f}"]
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("🎯 Risk Assessment")
    
    predict_btn = st.button("Predict Churn Risk", type="primary", use_container_width=True)
    
    if predict_btn and pipeline is not None:
        try:
            # Generate Prediction & Probabilities
            prediction = pipeline.predict(input_data)[0]
            probability = pipeline.predict_proba(input_data)[0][1]
            
            st.divider()
            
            # Metric Display
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Churn Probability", f"{probability * 100:.1f}%")
            with m2:
                status_label = "High Risk (Exit)" if prediction == 1 else "Low Risk (Retain)"
                st.metric("Status", status_label)
            
            st.progress(float(probability))
            
            # Risk Alert Display
            if prediction == 1:
                st.error(
                    f"⚠️ **High Churn Risk Detected!**\n\n"
                    f"This customer has a **{probability * 100:.1f}%** likelihood of leaving the bank. "
                    f"Consider targeted retention offers, personalized outreach, or account reviews."
                )
            else:
                st.success(
                    f"✅ **Low Churn Risk**\n\n"
                    f"This customer shows a **{(1 - probability) * 100:.1f}%** probability of remaining active."
                )
                
        except Exception as err:
            st.error(f"Failed to execute prediction: {err}")

# Footer Info
st.divider()
with st.expander("ℹ️ About the Model Pipeline"):
    st.write("""
    - **Preprocessing**: Handles categorical encoding (`country`, `gender`) via `OneHotEncoder` alongside standard numerical feature pass-through.
    - **Model**: `XGBClassifier` tuned for binary logloss prediction.
    """)
