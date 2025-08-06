# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('rf_churn_pipeline.pkl')

st.title("🔍 Customer Churn Prediction App")
st.write("Upload customer data to predict the probability of churn.")

# Upload data
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.subheader("📄 Input Data Preview")
    st.dataframe(input_df.head())

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]

    # Convert probabilities to percentage and round to 2 decimal places
    prediction_proba_percent = (prediction_proba * 100).round(2)

    # Show results
    st.subheader("🎯 Predictions")
    results = pd.DataFrame({
        'Churn Prediction': ['Yes' if x == 1 else 'No' for x in prediction],
        'Probability (%)': prediction_proba_percent
    })
    st.dataframe(results)

    st.success("✅ Prediction completed.")


