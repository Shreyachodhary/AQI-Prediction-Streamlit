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
# Model Training (Unchanged)
# ------------------------------
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

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(imputer, IMPUTER_PATH)

    return model, scaler, imputer

# ------------------------------
# Load Model with Caching (FIX APPLIED HERE)
# ------------------------------
@st.cache_resource(show_spinner=False)  # <--- THIS IS THE CRITICAL FIX
def load_objects():
    """Load model, scaler, imputer once and cache them."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        return model, scaler, imputer
    except FileNotFoundError:
        st.warning("⚠️ Model files not found. Please ensure PKL files are committed to GitHub or train the model below.")
        return None, None, None
    except Exception as e:
        # Catch version warnings or corrupt files
        st.error(f"An error occurred loading the model: {e}")
        return None, None, None

# ------------------------------
# Model Handling
# ------------------------------
# This line now only calls the function once per session due to the decorator.
model, scaler, imputer = load_objects()

uploaded_file = st.sidebar.file_uploader("📂 Upload your training CSV (must contain AQI + 8 features)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    missing = [col for col in FEATURES + ['AQI'] if col not in df.columns]
    if missing:
        st.sidebar.error(f"Missing columns in file: {missing}")
    else:
        if st.sidebar.button("Train Model"):
            with st.spinner("Training model..."):
                # If training occurs, clear the cache and rerun to load new model
                model, scaler, imputer = train_model(df)
            st.sidebar.success("✅ Model trained and saved successfully in this session!")
            st.cache_resource.clear()
            st.rerun()

# ------------------------------
# Prediction Section
# ------------------------------
st.subheader("🔮 Enter pollutant values to predict AQI")

# Input fields in two columns
col1, col2 = st.columns(2)
inputs = {}
for i, feat in enumerate(FEATURES):
    col = col1 if i < len(FEATURES)//2 else col2
    inputs[feat] = col.number_input(f"{feat}", min_value=0.0, format="%.3f")

if st.button("Predict AQI"):
    if model is None:
        st.error("⚠️ Please train the model first by uploading your dataset.")
    else:
        # Ensure preprocessing objects are available
        if imputer is None or scaler is None:
            st.error("⚠️ Preprocessing objects (imputer/scaler) are missing. Please train the model.")
        else:
            try:
                df_input = pd.DataFrame([inputs])
                df_imputed = imputer.transform(df_input)
                df_scaled = scaler.transform(df_imputed)
                pred = model.predict(df_scaled)[0]
                bucket = get_aqi_bucket(pred)

                st.success(f"### 🌎 Predicted AQI: {pred:.2f}")
                st.info(f"### Category: {bucket}")
            except Exception as e:
                st.error(f"Prediction failed. Please check your inputs or retrain the model. Error: {e}")

# ------------------------------
# Footer
# ------------------------------
st.write("---")
st.caption("Developed by Shreya 🌸 | Linear Regression AQI Predictor | Streamlit App")
