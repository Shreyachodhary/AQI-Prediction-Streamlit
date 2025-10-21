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
st.write("Enter pollutant concentrations to predict AQI and its category using a trained Linear Regression model.")

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
# Model Training Function
# ------------------------------
def train_model(df):
    """Train a Linear Regression model on the dataset and save preprocessing objects."""
    # Ensure columns exist
    missing = [col for col in FEATURES + ['AQI'] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only needed columns
    df = df[FEATURES + ['AQI']].copy()

    # Convert all columns to numeric (handles strings like 'NA' or '–')
    for col in FEATURES + ['AQI']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing AQI (target)
    df = df.dropna(subset=['AQI'])

    # If no valid data left, raise an error
    if df.empty:
        raise ValueError("No valid rows found after cleaning. Check your dataset.")

    # Separate features and target
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
# Load Model + Cache
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_objects():
    """Load model, scaler, imputer once and cache them."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        return model, scaler, imputer
    except FileNotFoundError:
        st.warning("⚠️ Model files not found. Please upload and train below.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# ------------------------------
# Load saved model if available
# ------------------------------
model, scaler, imputer = load_objects()

# ------------------------------
# Sidebar: Upload & Train
# ------------------------------
uploaded_file = st.sidebar.file_uploader("📂 Upload your training CSV (must contain AQI + 8 features)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    missing = [col for col in FEATURES + ['AQI'] if col not in df.columns]
    if missing:
        st.sidebar.error(f"Missing columns in file: {missing}")
    else:
        if st.sidebar.button("Train Model"):
            with st.spinner("Training model..."):
                model, scaler, imputer = train_model(df)
            st.sidebar.success("✅ Model trained and saved successfully!")
            st.cache_resource.clear()
            st.rerun()

# ------------------------------
# Prediction Section
# ------------------------------
st.subheader("🔮 Enter pollutant values to predict AQI")

col1, col2 = st.columns(2)
inputs = {}

for i, feat in enumerate(FEATURES):
    col = col1 if i < len(FEATURES)//2 else col2
    inputs[feat] = col.number_input(f"{feat}", min_value=0.0, format="%.3f")

if st.button("Predict AQI"):
    if model is None:
        st.error("⚠️ Please train the model first or ensure PKL files exist in repo.")
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
            st.error(f"Prediction failed. Error: {e}")

# ------------------------------
# Footer
# ------------------------------
st.write("---")
st.caption("Developed by Shreya 🌸 | Linear Regression AQI Predictor | Streamlit App")
