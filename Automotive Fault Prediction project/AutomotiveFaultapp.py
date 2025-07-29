import streamlit as st
import pandas as pd
import pickle
import os

st.title("🚗 Automotive Fault Prediction using ML")
st.markdown("Predict the type of automotive fault based on vehicle sensor readings.")

st.sidebar.header("Input Sensor Readings")

def user_input_features():
    speed = st.sidebar.slider('Speed (km/h)', 0, 200, 60)
    engine_temp = st.sidebar.slider('Engine Temperature (°C)', 40, 150, 90)
    oil_pressure = st.sidebar.slider('Oil Pressure (psi)', 10, 100, 50)
    vibration = st.sidebar.slider('Vibration Level', 0.0, 10.0, 5.0)
    brake_pressure = st.sidebar.slider('Brake Pressure (bar)', 0.0, 100.0, 50.0)
    gear_position = st.sidebar.slider('Gear Position', 1, 6, 3)
    data = {
        'speed': speed,
        'engine_temp': engine_temp,
        'oil_pressure': oil_pressure,
        'vibration': vibration,
        'brake_pressure': brake_pressure,
        'gear_position': gear_position
    }
    return pd.DataFrame([data])

input_df = user_input_features()

model_path = "rf_model.pkl"
if not os.path.exists(model_path):
    st.error("🚫 Trained model file 'rf_model.pkl' not found. Please train the model first.")
else:
    with open(model_path, "rb") as file:
        model_data = pickle.load(file)
    model = model_data["model"]
    label_encoder = model_data["label_encoder"]
    prediction = model.predict(input_df)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    st.subheader("Predicted Fault Type")
    st.success(f"🔧 {predicted_label}")
    st.subheader("Sensor Readings Provided")
    st.write(input_df)

