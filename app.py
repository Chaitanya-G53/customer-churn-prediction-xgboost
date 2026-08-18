import os
import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")


@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {e}")
        return None


model = load_model()

st.title("Customer Churn Prediction")
st.write("Enter customer metrics below to evaluate churn probability.")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=850, value=600
        )
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3)
        balance = st.number_input("Balance", min_value=0.0, value=60000.0)
        products_number = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
        credit_card = st.selectbox(
            "Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with col2:
        active_member = st.selectbox(
            "Is Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
        )
        estimated_salary = st.number_input(
            "Estimated Salary", min_value=0.0, value=50000.0
        )
        country = st.selectbox("Country", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])

    submit_button = st.form_submit_button("Predict Churn")

if submit_button:
    if model is None:
        st.error("Model file not found. Ensure `model.pkl` is in your repository root.")
    else:
        # Encode categorical variables to match training feature structure
        country_France = 1 if country == "France" else 0
        country_Germany = 1 if country == "Germany" else 0
        country_Spain = 1 if country == "Spain" else 0

        gender_Female = 1 if gender == "Female" else 0
        gender_Male = 1 if gender == "Male" else 0

        # Feature order matching the trained XGBoost model
        features = np.array(
            [
                [
                    credit_score,
                    age,
                    tenure,
                    balance,
                    products_number,
                    credit_card,
                    active_member,
                    estimated_salary,
                    country_France,
                    country_Germany,
                    country_Spain,
                    gender_Female,
                    gender_Male,
                ]
            ]
        )

        probability = float(model.predict_proba(features)[0][1])
        prediction = int(model.predict(features)[0])

        st.divider()
        if prediction == 1:
            st.error(f"**High Churn Risk** (Probability: {probability:.2%})")
        else:
            st.success(f"**Low Churn Risk** (Probability: {probability:.2%})")
