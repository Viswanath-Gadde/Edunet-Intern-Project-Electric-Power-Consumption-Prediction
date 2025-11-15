import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
# NOTE: The scaler is StandardScaler, fitted ONLY on the 5 physical/weather features.
# The model is MultiOutputRegressor(RandomForestRegressor).
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Error: The 'model.pkl' or 'scaler.pkl' files were not found. Please ensure they are in the same directory.")
    st.stop()


st.title("🔌 Power Consumption Predictor")

st.write("Enter the raw input values below:")

# ----------------------------
# 1. RAW FEATURES FROM USER
# ----------------------------
# Use more descriptive labels and initial values from the dataset snippet
dt = st.text_input("Datetime (YYYY-MM-DD HH:MM)", value="2017-01-01 00:00")
temperature = st.number_input("Temperature (°C)", value=6.559, format="%.3f")
humidity = st.number_input("Humidity (%)", value=73.8, format="%.3f")
windspeed = st.number_input("WindSpeed (m/s)", value=0.083, format="%.3f")
generaldiffuseflows = st.number_input("GeneralDiffuseFlows", value=0.051, format="%.3f")
diffuseflows = st.number_input("DiffuseFlows", value=0.119, format="%.3f")

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
        dayofweek = dt_parsed.dayofweek   # Monday=0
        isweekend = 1 if dayofweek >= 5 else 0

        # ----------------------------
        # 3. APPLY SCALING CORRECTLY
        # ----------------------------

        # Features to be scaled (5 features: Temp, Humidity, WindSpeed, GenDiff, Diff)
        # These features must be in the order the scaler was fitted on.
        weather_features_raw = np.array([
            temperature, humidity, windspeed,
            generaldiffuseflows, diffuseflows
        ]).reshape(1, -1)
        
        # Scale the weather features
        weather_features_scaled = scaler.transform(weather_features_raw)
        
        # Temporal features (6 features, which are NOT scaled)
        temporal_features = np.array([
            month, day, hour, minute, dayofweek, isweekend
        ]).reshape(1, -1)

        # ----------------------------
        # 4. COMBINE FINAL 11 FEATURES (IN ORDER)
        # ----------------------------
        # The final input for the model MUST match the training data structure:
        # [5 Scaled Weather Features | 6 Unscaled Temporal Features]
        X_final = np.hstack([weather_features_scaled, temporal_features])

        # Predict (The model expects 11 features in the correct order)
        predictions = model.predict(X_final)[0]
        
        st.success("🔮 Prediction Complete!")
        
        # Display results
        st.subheader("Predicted Power Consumption (kW)")
        st.write(f"**Zone 1:** **{predictions[0]:,.2f}**")
        st.write(f"**Zone 2:** {predictions[1]:,.2f}")
        st.write(f"**Zone 3:** {predictions[2]:,.2f}")


    except ValueError as ve:
        st.error(f"Input Error: Please check your date format (YYYY-MM-DD HH:MM) or input values. Details: {str(ve)}")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {str(e)}")