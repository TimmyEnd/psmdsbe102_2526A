import streamlit as st
import pandas as pd
import numpy as np

# Assume you already trained and saved your model + scaler + feature_columns
# For example, load them with joblib or pickle:
from joblib import load
final_model = load("final_model.pkl")
scaler = load("scaler.pkl")
feature_columns = load("feature_columns.pkl")

def predict_loan_application(model, scaler, application_data, feature_columns):
    """Predict loan default probability for a new application"""
    app_df = pd.DataFrame([application_data])
    for feature in feature_columns:
        if feature not in app_df.columns:
            app_df[feature] = 0
    app_df = app_df[feature_columns]
    app_scaled = scaler.transform(app_df)
    prediction = model.predict(app_scaled)[0]
    probability = model.predict_proba(app_scaled)[0, 1]
    return prediction, probability


# A list of options
options_list = ['Yes', 'No']


# ---------------- Streamlit UI ----------------
st.title("🚀 Loan Application Risk Predictor")
st.write("Enter applicant details to predict loan default probability.")

# Collect inputs
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=20000)
loan_int_rate = st.number_input("Loan Interest Rate in percent (annual)", value=10)
credit_score = st.number_input("Credit Score", min_value=300, max_value=2850, value=650)
person_emp_exp = st.number_input("Employment Length (years)", min_value=0, value=5)
#debt_to_income = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3)
previous_loan_defaults_on_file_Yes = st.selectbox('Defaulted on any previous loan?',options_list)
home_status = st.selectbox("Home Ownership", ["MORTGAGE", "OWN", "RENT"])

# Derived features
loan_percent_income = loan_amnt / person_income if person_income > 0 else 0
income_to_age = person_income / person_age if person_age > 0 else 0
credit_age_ratio = credit_score / person_age if person_age > 0 else 0

person_home_ownership_MORTGAGE = 1 if home_status == "MORTGAGE" else 0
person_home_ownership_OWN = 1 if home_status == "OWN" else 0
person_home_ownership_RENT = 1 if home_status == "RENT" else 0

previous_loan_defaults_string = previous_loan_defaults_on_file_Yes

previous_loan_defaults_on_file_Yes = 0 if previous_loan_defaults_on_file_Yes == "Yes" else 1.0

application_data = {
    "person_age": person_age,
    "person_income": person_income,
    "loan_int_rate": loan_int_rate,
    "loan_amnt": loan_amnt,
    "credit_score": credit_score,
    "person_emp_exp": person_emp_exp,
    #"debt_to_income": debt_to_income,
    "loan_percent_income": loan_percent_income,
    "income_to_age": income_to_age,
    "credit_age_ratio": credit_age_ratio,
    "person_home_ownership_MORTGAGE": person_home_ownership_MORTGAGE,
    "person_home_ownership_OWN": person_home_ownership_OWN,
    "person_home_ownership_RENT": person_home_ownership_RENT,
    "previous_loan_defaults_on_file_Yes": previous_loan_defaults_on_file_Yes,
}

if st.button("Predict Risk"):
    pred, prob = predict_loan_application(final_model, scaler, application_data, feature_columns)
    status = "✅ APPROVED" if pred == 0 else "❌ HIGH RISK"
    risk_level = "Low" if prob < 0.1 else "Medium" if prob < 0.6 else "High"

    st.subheader("📋 Prediction Result")
    st.write(f"**Status:** {status}")
    st.write(f"**Default Probability:** {prob:.1%} ({risk_level} Risk)")
    st.write(f"**Key Factors:** Credit Score {credit_score}, Income ${person_income:,.0f}, "
             f"Previous Loan Default {previous_loan_defaults_string}, Employment {person_emp_exp} years")
