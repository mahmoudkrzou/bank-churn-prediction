import streamlit as st
import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier, Pool

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "catboost_churn_model.cbm")

model = CatBoostClassifier()
model.load_model(MODEL_PATH)
THRESHOLD = 0.609

df = pd.read_csv('churn_dataset.csv')

df['last_transaction'] = pd.to_datetime(df['last_transaction'], errors='coerce')

df = df.dropna(subset=['last_transaction'])

num_cols = df.select_dtypes(include='number').columns
cat_cols = df.select_dtypes(include='object').columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna("Unknown")


def preprocess(raw_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    raw_df = raw_df.copy()
    ref_df = ref_df.copy()

    raw_df.drop(columns=['customer_id'], inplace=True)
    raw_df.drop(columns=['branch_code'], inplace=True)

    # Loyalty
    raw_df["loyalty"] = (
        (raw_df["vintage"] - ref_df["vintage"].min()) /
        (ref_df["vintage"].max() - ref_df["vintage"].min())
    )
    raw_df.drop(columns=['vintage'], inplace=True)

    # Engagement
    reference_date = ref_df['last_transaction'].max()

    ref_days = (reference_date - ref_df['last_transaction']).dt.days
    raw_days = (reference_date - raw_df['last_transaction']).dt.days

    raw_df['engagement'] = (
        raw_days - ref_days.min()
    ) / (ref_days.max() - ref_days.min())

    raw_df['engagement'] = 1 - raw_df['engagement']
    raw_df.drop(columns=['last_transaction'], inplace=True)


    # Gender
    raw_df['gender'] = raw_df['gender'].map({'Male': 0,'Female': 1, 'Unknown': 2}).fillna(2)

    # Balance features
    raw_df["balance_change"] = raw_df["current_month_balance"] - raw_df["previous_month_balance"]
    raw_df["balance_ratio"] = raw_df["current_month_balance"] / (raw_df["previous_month_balance"] + 1)

    raw_df["avg_balance_diff"] = (
        raw_df["average_monthly_balance_prevQ"] -
        raw_df["average_monthly_balance_prevQ2"]
    )
    raw_df["avg_balance_growth"] = (
        raw_df["avg_balance_diff"] /
        (raw_df["average_monthly_balance_prevQ2"] + 1)
    )

    raw_df["balance_volatility"] = raw_df[
        [
            "current_balance",
            'previous_month_balance',
            "previous_month_end_balance",
            "average_monthly_balance_prevQ",
            "average_monthly_balance_prevQ2",
        ]
    ].std(axis=1)

    # Credit / Debit
    raw_df["credit_change"] = raw_df["current_month_credit"] - raw_df["previous_month_credit"]
    raw_df["credit_ratio"] = raw_df["current_month_credit"] / (raw_df["previous_month_credit"] + 1)

    raw_df["debit_change"] = raw_df["current_month_debit"] - raw_df["previous_month_debit"]
    raw_df["debit_ratio"] = raw_df["current_month_debit"] / (raw_df["previous_month_debit"] + 1)

    raw_df.drop(columns=[
        'current_balance',
        'previous_month_end_balance',
        'average_monthly_balance_prevQ',
        'average_monthly_balance_prevQ2',
        'current_month_balance',
        'previous_month_balance',
        'current_month_credit',
        'previous_month_credit',
        'current_month_debit',
        'previous_month_debit'
    ], inplace=True)



    FEATURES = [
        "age",
        "gender",
        "dependents",
        "occupation",
        "city",
        "customer_nw_category",
        "loyalty",
        "engagement",
        "balance_change",
        "balance_ratio",
        "avg_balance_diff",
        "avg_balance_growth",
        "balance_volatility",
        "credit_change",
        "credit_ratio",
        "debit_change",
        "debit_ratio"
    ]

    return raw_df[FEATURES]

st.title("🏦 Bank Churn Prediction")

st.write("Enter **raw customer data** (before any processing)")

# Raw inputs
customer_id = st.number_input("Customer ID", value=1)
vintage = st.number_input("Vintage (months)", 0, 500, 60)
age = st.number_input("Age", 18, 100, 35)
gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
dependents = st.number_input("Dependents", 0, 10, 0)
occupation = st.selectbox("Occupation", ["self_employed", "salaried", "retired", "student", "company", "Unknown"])
city = st.text_input("City Code", "1020")
customer_nw_category = st.selectbox("Net Worth Category", ["1", "2", "3"])
branch_code = st.number_input("Branch Code", value=1)

current_balance = st.number_input("Current Balance", value=0.0)
previous_month_end_balance = st.number_input("Previous Month End Balance", value=0.0)
average_monthly_balance_prevQ = st.number_input("Avg Monthly Balance Prev Q", value=0.0)
average_monthly_balance_prevQ2 = st.number_input("Avg Monthly Balance Prev Q2", value=0.0)

current_month_credit = st.number_input("Current Month Credit", value=0.0)
previous_month_credit = st.number_input("Previous Month Credit", value=0.0)
current_month_debit = st.number_input("Current Month Debit", value=0.0)
previous_month_debit = st.number_input("Previous Month Debit", value=0.0)

current_month_balance = st.number_input("Current Month Balance", value=0.0)
previous_month_balance = st.number_input("Previous Month Balance", value=0.0)

last_transaction = st.date_input("Last Transaction Date")
last_transaction = pd.Timestamp(last_transaction)

if st.button("🔍 Predict Churn"):
    raw_df = pd.DataFrame([{
        "customer_id": customer_id,
        "vintage": vintage,
        "age": age,
        "gender": gender,
        "dependents": dependents,
        "occupation": occupation,
        "city": city,
        "customer_nw_category": customer_nw_category,
        "branch_code": branch_code,
        "current_balance": current_balance,
        "previous_month_end_balance": previous_month_end_balance,
        "average_monthly_balance_prevQ": average_monthly_balance_prevQ,
        "average_monthly_balance_prevQ2": average_monthly_balance_prevQ2,
        "current_month_credit": current_month_credit,
        "previous_month_credit": previous_month_credit,
        "current_month_debit": current_month_debit,
        "previous_month_debit": previous_month_debit,
        "current_month_balance": current_month_balance,
        "previous_month_balance": previous_month_balance,
        "last_transaction": last_transaction,
    }])

    X_processed = preprocess(raw_df, df)

    pool = Pool(
        X_processed,
        cat_features=["gender", "occupation", "city", "customer_nw_category"]
    )

    prob = model.predict_proba(pool)[0, 1]
    pred = int(prob >= THRESHOLD)

    st.subheader("📊 Prediction Result")
    st.write(f"**Churn Probability:** `{prob:.2f}`")

    if pred == 1:
        st.error("High Risk: Customer likely to churn")
    else:
        st.success("Low Risk: Customer likely to stay")
