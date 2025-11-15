import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sys

# --- Configuration ---
FEATURE_NAMES = [
    'Temperature', 'Humidity', 'WindSpeed', 
    'GeneralDiffuseFlows', 'DiffuseFlows',
    'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'IsWeekend'
]

# Load model and scaler
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    st.sidebar.success("Model and Scaler loaded successfully!")
except FileNotFoundError:
    st.error("Error: The 'model.pkl' or 'scaler.pkl' files were not found. Please ensure they are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading files: {e}")
    st.stop()

st.title("🔌 Power Consumption Predictor")

st.write("Enter the raw input values below:")

# ----------------------------
# 1. RAW FEATURES FROM USER INPUT
# ----------------------------

dt = st.text_input("Datetime (YYYY-MM-DD HH:MM)")

temperature = st.number_input("Temperature (°C)", format="%.3f")
humidity = st.number_input("Humidity (%)", format="%.3f")
windspeed = st.number_input("WindSpeed (m/s)", format="%.3f")
generaldiffuseflows = st.number_input("GeneralDiffuseFlows", format="%.3f")
diffuseflows = st.number_input("DiffuseFlows", format="%.3f")

if st.button("Predict"):
    try:
        # Convert to datetime
        dt_parsed = pd.to_datetime(dt)

        # ----------------------------
        # 2. CREATE 6 UNSCALED TEMPORAL FEATURES
        # ----------------------------
        month = dt_parsed.month
        day = dt_parsed.day
        hour = dt_parsed.hour
        minute = dt_parsed.minute
        dayofweek = dt_parsed.dayofweek
        isweekend = 1 if dayofweek >= 5 else 0

        # ----------------------------
        # 3. APPLY SCALING CORRECTLY
        # ----------------------------
        weather_features_raw = np.array([
            temperature, humidity, windspeed,
            generaldiffuseflows, diffuseflows
        ]).reshape(1, -1)

        weather_features_scaled = scaler.transform(weather_features_raw)

        temporal_features = np.array([
            month, day, hour, minute, dayofweek, isweekend
        ]).reshape(1, -1)

        # ----------------------------
        # 4. COMBINE FINAL 11 FEATURES
        # ----------------------------
        X_numpy = np.hstack([weather_features_scaled, temporal_features])
        X_final = pd.DataFrame(X_numpy, columns=FEATURE_NAMES)

        predictions = model.predict(X_final)[0]

        st.success("🔮 Prediction Complete!")

        st.subheader("Predicted Power Consumption (kW)")
        st.write(f"**Zone 1:** **{predictions[0]:,.2f}**")
        st.write(f"**Zone 2:** {predictions[1]:,.2f}")
        st.write(f"**Zone 3:** {predictions[2]:,.2f}")

    except ValueError as ve:
        st.error(f"Input Error: Please check your date format (YYYY-MM-DD HH:MM). Details: {str(ve)}")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {str(e)}. Please confirm your model/scaler files.")
