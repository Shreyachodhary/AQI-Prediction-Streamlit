import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ------------------------------
# Configuration
# ------------------------------
FEATURES = ['PM2.5', 'PM10', 'NO2', 'NH3', 'CO', 'SO2', 'O3', 'Benzene']
MODEL_PATH = "linear_aqi_model.pkl"
SCALER_PATH = "linear_scaler.pkl"
IMPUTER_PATH = "linear_imputer.pkl"

st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="wide")
st.title("🌫️ Air Quality Index (AQI) Predictor")
st.write("Enter pollutant concentrations to predict AQI and its category using your trained Linear Regression model.")

# ------------------------------
# AQI Bucket Function
# ------------------------------
def get_aqi_bucket(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# ------------------------------
# Load Model with Caching (The key change!)
# ------------------------------
@st.cache_resource
def load_objects():
    """Load model, scaler, imputer once and cache them."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        # Check to ensure all objects were loaded successfully
        if model and scaler and imputer:
            return model, scaler, imputer
        else:
            # If a file exists but is corrupt, handle it gracefully
            return None, None, None
    except FileNotFoundError:
        # This is the expected state if the files haven't been uploaded to GitHub yet
        return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during model loading: {e}")
        return None, None, None

# ------------------------------
# Model Training (Only runs if a user explicitly clicks 'Train Model')
# ------------------------------
# You should NOT cache this function, as it needs to run every time the user trains.
def train_model(df):
    """Train linear regression on the dataset and save preprocessing objects"""
    X = df[FEATURES]
    y = df['AQI']

    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    model = LinearRegression()
    model.fit(X_scaled, y)

    # Note: These files are saved in the temporary container storage.
    # To persist them, you must commit them to GitHub after training locally.
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(imputer, IMPUTER_PATH)

    return model, scaler, imputer

# ------------------------------
# Model Handling
# ------------------------------
# The cache decorator ensures load_objects() only runs once per app deployment.
model, scaler, imputer = load_objects()

uploaded_file = st.sidebar.file_uploader("📂 Upload your training CSV (must contain AQI + 8 features)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    missing = [col for col in FEATURES + ['AQI'] if col not in df.columns]
    if missing:
        st.sidebar.error(f"Missing columns in file: {missing}")
    else:
        # IMPORTANT: The model will only be saved in the temporary deployment environment.
        # To truly persist it, you need to train locally and upload the .pkl files to GitHub.
        if st.sidebar.button("Train Model (Temporary)"):
            with st.spinner("Training model..."):
                model, scaler, imputer = train_model(df)
            st.sidebar.success("✅ Model trained and saved successfully in this session!")
            # Manually clear the resource cache to force the new model to be loaded
            st.cache_resource.clear() 
            st.rerun() # Rerun the app to reload with the new objects

# ------------------------------
# Prediction Section
# ------------------------------
# ... (rest of the prediction code remains the same)
# ...
if st.button("Predict AQI"):
    # ... (prediction logic)
# ... (prediction logic)
# ------------------------------
# Footer
# ------------------------------
st.write("---")
st.caption("Developed by Shreya 🌸 | Linear Regression AQI Predictor | Streamlit App")
