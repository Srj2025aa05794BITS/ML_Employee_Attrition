import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Employee Attrition Model Evaluation")

# ---------------- MODEL SELECTION ----------------
model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression","Decision Tree","KNN","Naive Bayes","Random Forest","XGBoost"]
)

model_file = model_name.lower().replace(" ", "_") + "_model.pkl"
model = pickle.load(open(f"models/{model_file}", "rb"))

# ---------------- DATA UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.write(df.head())

    # dataset already preprocessed in notebook
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    X_scaled = X.values

    # ---------------- PREDICTION ----------------
    y_pred = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled)[:,1]
        auc = roc_auc_score(y, y_prob)
    else:
        auc = 0

    # ---------------- METRICS ----------------
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    st.subheader("Evaluation Metrics")

    st.write(f"Accuracy: {acc:.3f}")
    st.write(f"Precision: {pre
