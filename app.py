import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load("rf_churn_pipeline.pkl")

st.title("🔍 Customer Churn Prediction App")
st.write("Enter customer details to predict churn probability and get recommendations.")

# Form for user input
with st.form("churn_form"):
    # Numerical inputs
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=840.0)

    # Categorical inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    submitted = st.form_submit_button("Predict Churn")

# On submit
if submitted:
    # Create input DataFrame
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Predictions
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    # Display results
    st.subheader("📊 Prediction Result")
    st.write(f"**Churn:** {'Yes' if prediction == 1 else 'No'}")
    st.write(f"**Churn Probability:** {probability:.2%}")

    # Recommendation logic
    st.subheader("💡 Recommendation")
    if probability >= 0.7:
        st.error("High churn risk — Offer immediate retention incentives like discounts or service upgrades.")
    elif 0.4 <= probability < 0.7:
        st.warning("Moderate churn risk — Engage with personalized offers and proactive support.")
    else:
        st.success("Low churn risk — Maintain customer satisfaction through consistent service quality.")
