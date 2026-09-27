"""
Car Selling Price Predictor — Streamlit App
Section 8: Model Deployment with Web App

Replicates the EXACT preprocessing pipeline used in the notebook, in order:
  1. log1p transform on engine, max_power, km_driven      (notebook cell 35)
  2. Global StandardScaler on 6 numeric cols               (notebook cell 37, pre-split)
  3. Binary encode transmission_type                       (notebook cell 59)
  4. Frequency-encode brand / model                        (notebook cell 59)
  5. One-hot encode fuel_type / seller_type                (notebook cell 59)
  6. Second StandardScaler on the SAME 6 numeric cols       (notebook cell 59, train-only fit)
  7. Align columns to training order, then predict with best_rf

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------------------------
# Load model + all preprocessing artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_model.pkl")
    scaler_global = joblib.load("scaler_global.pkl")      # cell 37 scaler
    scaler_trainonly = joblib.load("scaler_trainonly.pkl")  # cell 59 scaler
    encoder = joblib.load("onehot_encoder.pkl")
    brand_freq_map = joblib.load("brand_freq_map.pkl")
    model_freq_map = joblib.load("model_freq_map.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return (model, scaler_global, scaler_trainonly, encoder,
            brand_freq_map, model_freq_map, feature_columns)


(model, scaler_global, scaler_trainonly, encoder,
 brand_freq_map, model_freq_map, feature_columns) = load_artifacts()

# Columns scaled TWICE in the notebook (cell 37, then again in cell 59) — same order both times
NUMERIC_FEATURES_TO_SCALE = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]

# Columns that get log1p-transformed BEFORE any scaling (notebook cell 35)
LOG_TRANSFORM_COLS = ["engine", "max_power", "km_driven"]

ONEHOT_COLS = ["fuel_type", "seller_type"]

st.set_page_config(page_title="Car Price Predictor", page_icon="🚙✨", layout="centered")

st.title("🚙✨ Car Price Predictor")
st.write(
    "Enter the details of a used car below to get a Fair selling price, "
    
)
st.divider()

# ---------------------------------------------------------------------------
# Collect raw, human-readable input — units as they'd appear on a listing
# ---------------------------------------------------------------------------
st.subheader("Your Car Details")

col1, col2 = st.columns(2)

with col1:
    brand = st.text_input("Brand (e.g. Maruti, Hyundai, Honda)", value="Please enter your car Brand")
    model_name = st.text_input("Model (e.g. Swift, i20, City)", value="Please enter your car Model")
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=3)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500_000, value=10_000, step=1000)
    mileage = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)

with col2:
    engine = st.number_input("Engine Capacity (cc)", min_value=500, max_value=6000, value=800, step=50)
    max_power = st.number_input("Max Power (bhp)", min_value=20.0, max_value=600.0, value=50.0, step=1.0)
    seats = st.number_input("Number of Seats", min_value=2, max_value=14, value=5)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission_type = st.selectbox("Transmission", ["Manual", "Automatic"])

st.caption(
    "⚠️ Confirm these dropdown values exactly match the categories in your dataset "
    "(run `Car_data['fuel_type'].unique()` etc. in the notebook to check) — mismatches "
    "will be treated as an unseen category and encoded as all-zeros."
)

predict_clicked = st.button("Predict Selling Price", type="primary")


# ---------------------------------------------------------------------------
# Preprocessing — replicates cells 35, 37, and 59 in exact order, transform-only
# ---------------------------------------------------------------------------
def preprocess_input(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    # 1. Log-transform engine, max_power, km_driven (notebook cell 35)
    for col in LOG_TRANSFORM_COLS:
        df[col] = np.log1p(df[col])

    # 2. Global scaler — first pass (notebook cell 37)
    df[NUMERIC_FEATURES_TO_SCALE] = scaler_global.transform(df[NUMERIC_FEATURES_TO_SCALE])

    # 3. Binary encode transmission_type (notebook cell 59)
    df["transmission_type"] = df["transmission_type"].map({"Manual": 0, "Automatic": 1})

    # 4. Frequency-encode brand / model using saved training-set frequency maps
    df["brand_freq"] = df["brand"].map(brand_freq_map).fillna(0)
    df["model_freq"] = df["model"].map(model_freq_map).fillna(0)
    df = df.drop(columns=["brand", "model"])

    # 5. One-hot encode fuel_type / seller_type using the saved, already-fitted encoder
    encoded = encoder.transform(df[ONEHOT_COLS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(ONEHOT_COLS),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=ONEHOT_COLS), encoded_df], axis=1)

    # 6. Second scaler — train-only pass, applied AGAIN to the same 6 columns (notebook cell 59)
    df[NUMERIC_FEATURES_TO_SCALE] = scaler_trainonly.transform(df[NUMERIC_FEATURES_TO_SCALE])

    # 7. Align to the exact column set/order the model was trained on
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if predict_clicked:
    raw_input = {
        "brand": brand.strip(),
        "model": model_name.strip(),
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "fuel_type": fuel_type,
        "seller_type": seller_type,
        "transmission_type": transmission_type,
    }

    try:
        X_new = preprocess_input(raw_input)
        prediction = model.predict(X_new)[0]

        st.divider()
        st.subheader("Predicted Selling Price")
        st.metric(label="Estimated Price", value=f"₹{prediction:,.0f}")

        st.caption(
            "Based on a Random Forest model with a test-set R² of ~0.89 and an average "
            "error (MAE) of roughly ₹65,700. Actual prices may vary based on factors not "
            "captured by the model (condition, accident history, negotiation, etc.)."
        )
    except Exception as e:
        st.error(f"Something went wrong while generating the prediction: {e}")

st.divider()
st.caption("Model: Random Forest Regressor · Trained on historical used-car listing data")
