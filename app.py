import streamlit as st
import pickle
import numpy as np

st.title("Employee Attrition Prediction")

# choose model
model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression","Decision Tree","KNN","Naive Bayes","Random Forest","XGBoost"]
)

model_file = model_name.lower().replace(" ", "_") + "_model.pkl"
model = pickle.load(open(f"models/{model_file}","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

age = st.slider("Age",18,60,30)
income = st.number_input("Monthly Income",1000,50000,10000)

sample = np.zeros((1,30))
sample[0,0] = age
sample[0,1] = income

sample = scaler.transform(sample)

if st.button("Predict"):
    pred = model.predict(sample)

    if pred[0]==1:
        st.error("Employee Likely to Leave")
    else:
        st.success("Employee Likely to Stay")
