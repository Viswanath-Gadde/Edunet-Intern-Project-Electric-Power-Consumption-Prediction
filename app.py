import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Correct Feature Order (NO SCALING) ---
FEATURE_NAMES = [
    'Temperature', 'Humidity', 'WindSpeed', 
    'GeneralDiffuseFlows', 'DiffuseFlows',
    'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'IsWeekend'
]

# Load model (NO SCALER NEEDED)
try:
    model = joblib.load("model.pkl")
    st.sidebar.success("Model loaded successfully!")
except:
    st.error("model.pkl not found!")
    st.stop()


st.title("🔌 Power Consumption Predictor (No Scaling)")

dt = st.text_input("Datetime (YYYY-MM-DD HH:MM)")

temperature = st.number_input("Temperature (°C)", format="%.3f")
humidity = st.number_input("Humidity (%)", format="%.3f")
windspeed = st.number_input("WindSpeed (m/s)", format="%.3f")
generaldiffuseflows = st.number_input("GeneralDiffuseFlows", format="%.3f")
diffuseflows = st.number_input("DiffuseFlows", format="%.3f")

if st.button("Predict"):

    try:
        # Parse datetime
        dt_parsed = pd.to_datetime(dt)

        # Temporal features (NOT scaled)
        month = dt_parsed.month
        day = dt_parsed.day
        hour = dt_parsed.hour
        minute = dt_parsed.minute
        dayofweek = dt_parsed.dayofweek
        isweekend = 1 if dayofweek >= 5 else 0

        # Create final input (NO SCALING)
        X = pd.DataFrame([[
            temperature, humidity, windspeed,
            generaldiffuseflows, diffuseflows,
            month, day, hour, minute, dayofweek, isweekend
        ]], columns=FEATURE_NAMES)

        # Predict
        predictions = model.predict(X)[0]

        st.success("🔮 Prediction Complete!")
        st.subheader("Predicted Power Consumption (kW)")
        st.write(f"**Zone 1:** {predictions[0]:,.2f}")
        st.write(f"**Zone 2:** {predictions[1]:,.2f}")
        st.write(f"**Zone 3:** {predictions[2]:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
