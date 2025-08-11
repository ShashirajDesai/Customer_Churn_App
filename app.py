import streamlit as st
import pandas as pd
import joblib
import shap

# Load trained pipeline
pipeline = joblib.load("rf_churn_pipeline.pkl")

# Load SHAP explainer for model
explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])

st.title("🔍 Customer Churn Prediction App")
st.write("Enter customer details below to predict churn probability and see key influencing factors.")

# Form for user input
with st.form("churn_form"):
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=840.0)

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

if submitted:
    # Create single-row DataFrame
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

    # Prediction
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")
    st.write(f"**Churn:** {'Yes' if prediction == 1 else 'No'}")
    st.write(f"**Churn Probability:** {probability:.2%}")

    # Recommendation
    st.subheader("💡 Recommendation")
    if probability > 0.7:
        st.warning("High churn risk! Consider offering discounts or upgrading customer support.")
    elif probability > 0.4:
        st.info("Moderate churn risk. Engage customer with loyalty rewards or personalized offers.")
    else:
        st.success("Low churn risk. Maintain good service quality.")

    # SHAP explanation
    st.subheader("🔍 Top Factors Influencing Prediction")
    
    # Preprocess input for SHAP
    X_processed = pipeline.named_steps['preprocessor'].transform(input_data)
    
    # Get SHAP values
    shap_values = explainer.shap_values(X_processed)
    
    # Get feature names after encoding
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    # Map feature importances for this prediction
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values[1][0]  # Class 1 = Churn
    })
    feature_importance = feature_importance.reindex(feature_importance["SHAP Value"].abs().sort_values(ascending=False).index)
    
    st.table(feature_importance.head(3))
