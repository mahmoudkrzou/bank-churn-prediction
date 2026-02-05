# Bank Churn Prediction App

This project is a **Streamlit web app** that predicts **bank customer churn** using a **CatBoost model**. Users can input raw customer data, and the app will process it, compute derived features, and predict the probability of churn.

---

## Features

- Accepts **raw customer data** before preprocessing.
- Handles **missing values** and feature engineering automatically.
- Computes **loyalty**, **engagement**, **balance**, **credit**, and **debit features**.
- Predicts churn probability using a **CatBoostClassifier**.
- Highlights whether a customer is **likely to churn**.

---

## Installation

1. Clone the repository:

 - git clone https://github.com/mahmoudkrzou/bank-churn-prediction.git
 - cd bank-churn-prediction

2. Create Create a virtual environment:

 - python3 -m venv venv
 - source venv/bin/activate

3. Install dependencies:

 - pip install -r requirements.txt
 
---

## Usage

Run the Streamlit app:

 - streamlit run app.py

Fill in the customer data, and click Predict Churn to see the results.

---

## Project Structure

bank-churn-prediction/
├─ app.py                  # Streamlit app
├─ catboost_churn_model.cbm # Trained CatBoost model
├─ churn_dataset.csv       # Original dataset
├─ churnprediction.ipynb   # Jupyter notebook with EDA & model training
├─ requirements.txt        # Python dependencies
└─ README.md               # Project description

---

## Notes

The app expects raw features as input and applies the same preprocessing used during model training.
Churn threshold is set at 0.61 (can be adjusted in app.py).

---
