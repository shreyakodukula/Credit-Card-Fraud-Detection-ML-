import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load("model.pkl")  # make sure you save your model in repo

st.title("💳 Credit Card Fraud Detection Demo")

st.write("Enter transaction details to check if it's fraudulent:")

# Example input fields (modify based on your dataset features)
amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, value=1000.0)
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, value=900.0)

# Put features into DataFrame (adjust feature names to match your training data)
features = pd.DataFrame([[amount, oldbalanceOrg, newbalanceOrig]],
                        columns=["amount", "oldbalanceOrg", "newbalanceOrig"])

if st.button("Predict"):
    prediction = model.predict(features)[0]
    result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legit Transaction"
    st.subheader(result)

