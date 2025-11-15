import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# --- Configuration ---
# CRITICAL: These are the exact 11 feature names and order required by the model.
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


st.title("Power Consumption Predictor")

st.write("Enter the raw input values below:")

# ----------------------------
# 1. RAW FEATURES FROM USER INPUT (Updated Datetime Input)
# ----------------------------

# Separate Date and Time inputs for better user experience
st.subheader("Date & Time Input")
col1, col2 = st.columns(2)
with col1:
    input_date = st.date_input("Date", value=datetime(2017, 1, 1).date())
with col2:
    input_time = st.time_input("Time", value=datetime(2017, 0, 1, 0, 0).time())

# Combine date and time into a single datetime object
dt_parsed = datetime.combine(input_date, input_time)


st.subheader("Weather Features")
temperature = st.number_input("Temperature (°C)", value=6.559, format="%.3f")
humidity = st.number_input("Humidity (%)", value=73.8, format="%.3f")
windspeed = st.number_input("WindSpeed (m/s)", value=0.083, format="%.3f")
generaldiffuseflows = st.number_input("GeneralDiffuseFlows", value=0.051, format="%.3f")
diffuseflows = st.number_input("DiffuseFlows", value=0.119, format="%.3f")


if st.button("Predict"):
    try:
        # We already have the datetime object: dt_parsed

        # ----------------------------
        # 2. CREATE 6 UNSCALED TEMPORAL FEATURES
        # ----------------------------
        month = dt_parsed.month
        day = dt_parsed.day
        hour = dt_parsed.hour
        minute = dt_parsed.minute
        dayofweek = dt_parsed.weekday()   # Monday=0 (Same as dt_parsed.dayofweek)
        isweekend = 1 if dayofweek >= 5 else 0

        # ----------------------------
        # 3. APPLY SCALING CORRECTLY
        # ----------------------------

        # 5 features to be scaled (MUST be in the correct order: T, H, WS, GDF, DF)
        weather_features_raw = np.array([
            temperature, humidity, windspeed,
            generaldiffuseflows, diffuseflows
        ]).reshape(1, -1)
        
        # Scale the weather features
        weather_features_scaled = scaler.transform(weather_features_raw)
        
        # 6 Temporal features (NOT scaled)
        temporal_features = np.array([
            month, day, hour, minute, dayofweek, isweekend
        ]).reshape(1, -1)

        # ----------------------------
        # 4. COMBINE FINAL 11 FEATURES (AS DATAFRAME)
        # ----------------------------
        # Order MUST be: [5 Scaled Weather | 6 Unscaled Temporal]
        X_numpy = np.hstack([weather_features_scaled, temporal_features])

        # CRITICAL FIX: Convert to DataFrame with feature names
        X_final = pd.DataFrame(X_numpy, columns=FEATURE_NAMES)

        # Predict (The model expects the 11-feature DataFrame)
        predictions = model.predict(X_final)[0]
        
        st.success("🔮 Prediction Complete!")
        
        # Display results
        st.subheader("Predicted Power Consumption (kW)")
        st.write(f"**Zone 1:** **{predictions[0]:,.2f}**")
        st.write(f"**Zone 2:** {predictions[1]:,.2f}")
        st.write(f"**Zone 3:** {predictions[2]:,.2f}")


    except ValueError as ve:
        st.error(f"Input Error: Please check your input values. Details: {str(ve)}")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {str(e)}. Please confirm your model/scaler files.")