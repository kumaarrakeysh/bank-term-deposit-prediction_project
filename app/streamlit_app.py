import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/best_model.pkl")

st.set_page_config(
    page_title="Bank Deposit Prediction",
    layout="centered"
)

st.title("🏦 Bank Term Deposit Prediction")

st.write(
    "Predict whether customer will subscribe "
    "to a term deposit."
)

# -----------------------------
# USER INPUTS
# -----------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

balance = st.number_input(
    "Balance",
    value=1000
)

duration = st.number_input(
    "Call Duration",
    min_value=0,
    value=100
)

campaign = st.number_input(
    "Campaign Contacts",
    min_value=1,
    value=1
)

job = st.selectbox(
    "Job",
    [
        'admin.',
        'technician',
        'services',
        'management',
        'blue-collar',
        'student',
        'retired'
    ]
)

marital = st.selectbox(
    "Marital Status",
    ['married', 'single', 'divorced']
)

education = st.selectbox(
    "Education",
    ['primary', 'secondary', 'tertiary']
)

housing = st.selectbox(
    "Housing Loan",
    ['yes', 'no']
)

loan = st.selectbox(
    "Personal Loan",
    ['yes', 'no']
)

contact = st.selectbox(
    "Contact Type",
    ['cellular', 'telephone']
)

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Predict"):

    sample = pd.DataFrame([{

        'age': age,
        'balance': balance,
        'duration': duration,
        'campaign': campaign,
        'job': job,
        'marital': marital,
        'education': education,
        'housing': housing,
        'loan': loan,
        'contact': contact,

        # Default Values
        'default': 'no',
        'day': 5,
        'month': 'may',
        'pdays': -1,
        'previous': 0,
        'poutcome': 'unknown',

        # Engineered Features
        'age_balance': age * balance,
        'campaign_duration': campaign * duration
    }])

    prediction = model.predict(sample)[0]

    probability = model.predict_proba(sample)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            f"Likely to Subscribe "
            f"(Probability: {probability:.2%})"
        )
    else:
        st.error(
            f"Not Likely to Subscribe "
            f"(Probability: {probability:.2%})"
        )