# 🧠 Customer Churn Prediction App (Streamlit)

This Streamlit app predicts whether a customer is likely to churn based on input features. It uses a Random Forest model wrapped in a preprocessing pipeline trained on the Telco Customer Churn dataset.

---

## 🚀 Features

- Upload raw customer data as CSV
- Automatic preprocessing (encoding + scaling)
- Real-time churn prediction with probabilities
- Easy web deployment with Streamlit Cloud

---

## 📁 Files

- `app.py`: Main Streamlit application
- `rf_churn_pipeline.pkl`: Trained pipeline (preprocessing + model)
- `input_sample.csv`: Sample input for testing
- `requirements.txt`: Python dependencies
- `README.md`: Project overview

---

## 📦 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/churn-prediction-app.git
cd churn-prediction-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run locally

```bash
streamlit run app.py
```

---

## ☁️ Streamlit Cloud Deployment

1. Push this project to a public GitHub repository  
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)  
3. Connect your repo and deploy with default settings

---

## 📊 Dataset Source

- [Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 📬 Contact

Feel free to reach out for feedback or collaboration!