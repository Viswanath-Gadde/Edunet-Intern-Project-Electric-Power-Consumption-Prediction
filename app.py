import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔌 Power Consumption Predictor")

st.write("Enter the raw input values below:")

# ----------------------------
# 1. RAW FEATURES FROM USER
# ----------------------------
dt = st.text_input("Datetime (YYYY-MM-DD HH:MM)")
temperature = st.number_input("Temperature", value=0.0)
humidity = st.number_input("Humidity", value=0.0)
windspeed = st.number_input("WindSpeed", value=0.0)
generaldiffuseflows = st.number_input("GeneralDiffuseFlows", value=0.0)
diffuseflows = st.number_input("DiffuseFlows", value=0.0)

if st.button("Predict"):
    try:
        # Convert to datetime
        dt_parsed = pd.to_datetime(dt)

        # ----------------------------
        # 2. CREATE 6 TEMPORAL FEATURES
        # ----------------------------
        month = dt_parsed.month
        day = dt_parsed.day
        hour = dt_parsed.hour
        minute = dt_parsed.minute
        dayofweek = dt_parsed.dayofweek   # Monday=0
        isweekend = 1 if dayofweek >= 5 else 0

        # ----------------------------
        # 3. FINAL 11 FEATURES (IN ORDER)
        # ----------------------------
        features = [
            month, day, hour, minute, dayofweek, isweekend,
            temperature, humidity, windspeed,
            generaldiffuseflows, diffuseflows
        ]

        X = np.array(features).reshape(1, -1)

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        prediction = model.predict(X_scaled)[0]

        st.success(f"🔮 Predicted Power Consumption: **{prediction}**")

    except Exception as e:
        st.error(f"Error: {str(e)}")
