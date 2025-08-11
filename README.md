# 🧠 Customer Churn Prediction App (Streamlit)

This Streamlit app predicts whether a customer is likely to churn based on input features.  
It uses a **Random Forest model** wrapped in a **preprocessing pipeline** trained on the Telco Customer Churn dataset.

Now enhanced with:
- **Manual customer detail entry** — no need to upload a CSV
- **Automated churn probability calculation**
- **Personalized business recommendations** based on churn risk

---

## 🚀 Features

- **Two Modes of Prediction**:
  1. **Manual Input Mode** — Enter new customer details through a simple form
  2. **Bulk CSV Upload** — Predict churn for multiple customers at once
- Automatic preprocessing (encoding + scaling) — no manual data cleaning required
- Real-time churn prediction with probabilities
- Actionable recommendations for retention strategies
- Easy web deployment with Streamlit Cloud

---

## 📁 Files

- `app.py` → Main Streamlit application
- `rf_churn_pipeline.pkl` → Trained pipeline (preprocessing + model)
- `input_sample.csv` → Sample input for testing
- `requirements.txt` → Python dependencies
- `README.md` → Project overview and instructions

---

## 📦 Setup Instructions

### 1️⃣ Clone the repo
```bash
git clone https://github.com/your-username/churn-prediction-app.git
cd churn-prediction-app
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run locally

```bash
streamlit run app.py
```

---

## ☁️ Streamlit Cloud Deployment

1. Push this project to a **public GitHub repository**
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repo and deploy with default settings

---

## 🖥 How to Use

### **Manual Input Mode**

1. Open the app
2. Enter customer details in the provided form
3. Click **Predict Churn**
4. View the churn probability and a **personalized recommendation**

### **CSV Upload Mode**

1. Prepare a CSV with the same structure as `input_sample.csv`
2. Upload it through the app
3. View predictions for all customers

---

## 📊 Dataset Source

* [Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 📬 Contact

Feel free to reach out for feedback or collaboration!

```

---

If you want, I can save this as a new **README.md** so you can **replace your old one** in the GitHub repo directly.  

Do you want me to overwrite the existing `README.md` with this updated version?
```
