import joblib
import numpy as np
import pandas as pd
import os

# --- Configuration ---
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# CRITICAL: These are the exact 11 feature names and order required by the model.
FEATURE_NAMES = [
    'Temperature', 'Humidity', 'WindSpeed', 
    'GeneralDiffuseFlows', 'DiffuseFlows',
    'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'IsWeekend'
]

def load_assets():
    """Loads the model and scaler objects."""
    print("--- Loading Assets ---")
    try:
        # Load the model and scaler
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and Scaler loaded successfully.")
        return model, scaler
    except FileNotFoundError:
        print(f"ERROR: Could not find {MODEL_PATH} or {SCALER_PATH}.")
        print("Please ensure both files are in the same directory as this script.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        exit(1)

def get_user_input():
    """Collects raw feature data from the user."""
    print("\n--- Enter Prediction Data (Example: 2017-01-01 00:00) ---")
    
    # Use default values for easy testing
    dt = input("Datetime (YYYY-MM-DD HH:MM): ") or "2017-01-01 00:00"
    temperature = float(input("Temperature (°C): ") or 6.559)
    humidity = float(input("Humidity (%): ") or 73.8)
    windspeed = float(input("WindSpeed (m/s): ") or 0.083)
    generaldiffuseflows = float(input("GeneralDiffuseFlows: ") or 0.051)
    diffuseflows = float(input("DiffuseFlows: ") or 0.119)
    
    return dt, temperature, humidity, windspeed, generaldiffuseflows, diffuseflows

def predict_power_consumption(model, scaler, raw_inputs):
    """
    Processes raw inputs, applies scaling, and generates a prediction.
    """
    dt, temperature, humidity, windspeed, generaldiffuseflows, diffuseflows = raw_inputs

    try:
        # 1. TEMPORAL FEATURES (UNSCALED)
        dt_parsed = pd.to_datetime(dt)
        month = dt_parsed.month
        day = dt_parsed.day
        hour = dt_parsed.hour
        minute = dt_parsed.minute
        dayofweek = dt_parsed.dayofweek
        isweekend = 1 if dayofweek >= 5 else 0

        temporal_features = np.array([month, day, hour, minute, dayofweek, isweekend]).reshape(1, -1)
        
        # 2. WEATHER FEATURES (SCALED)
        # MUST be in the correct order: T, H, WS, GDF, DF
        weather_features_raw = np.array([
            temperature, humidity, windspeed,
            generaldiffuseflows, diffuseflows
        ]).reshape(1, -1)
        
        # Apply scaling to the 5 weather features
        weather_features_scaled = scaler.transform(weather_features_raw)
        
        # 3. COMBINE FINAL 11 FEATURES (AS NUMPY ARRAY)
        # Order MUST be: [5 Scaled Weather | 6 Unscaled Temporal]
        X_numpy = np.hstack([weather_features_scaled, temporal_features])

        # 4. CRITICAL FIX: CONVERT TO DATAFRAME WITH FEATURE NAMES
        # This addresses the 'only length-1 arrays can be converted to Python scalars' error.
        X_final = pd.DataFrame(X_numpy, columns=FEATURE_NAMES)
        
        # Print debug info to console
        print("\n--- Prediction Array Details ---")
        print(f"Shape of X_final (must be (1, 11)): {X_final.shape}")
        print("Final 11 Features for Model Input (DataFrame):")
        print(X_final.to_string(index=False))
        print("---")

        # 5. PREDICT
        # The model now receives the expected DataFrame format.
        predictions = model.predict(X_final)[0]
        
        return predictions

    except ValueError as ve:
        print(f"\nERROR: Input format issue: {ve}")
        print("Please check your date format (YYYY-MM-DD HH:MM) and ensure numeric inputs are valid.")
        return None
    except Exception as e:
        # Catch any errors from the prediction step itself
        print(f"\nAN UNEXPECTED MODEL ERROR OCCURRED: {e}")
        print("If the error persists, there may be a mismatch between the loaded model and feature set.")
        return None


if __name__ == "__main__":
    model, scaler = load_assets()
    
    raw_inputs = get_user_input()
    
    predictions = predict_power_consumption(model, scaler, raw_inputs)
    
    if predictions is not None:
        print("\n--- PREDICTION RESULTS (kW) ---")
        print(f"Zone 1: {predictions[0]:,.2f}")
        print(f"Zone 2: {predictions[1]:,.2f}")
        print(f"Zone 3: {predictions[2]:,.2f}")
        print("-----------------------------\n")
        