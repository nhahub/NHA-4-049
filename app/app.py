import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('../models/model.pkl')
model_columns = joblib.load('../models/model_columns.pkl')

st.title("Predictive Maintenance System")
st.write("Enter machine sensor readings to predict if maintenance is needed within 7 days.")

# --- User Inputs ---
st.subheader("Machine Type")
machine_type = st.selectbox("Select Machine Type", [
    "Automated_Screwdriver", "AGV", "Boiler", "CMM", "CNC_Lathe",
    "CNC_Mill", "Carton_Former", "Compressor", "Conveyor_Belt", "Crane",
    "Dryer", "Forklift_Electric", "Furnace", "Grinder", "Heat_Exchanger",
    "Hydraulic_Press", "Industrial_Chiller", "Injection_Molder", "Labeler",
    "Laser_Cutter", "Mixer", "Palletizer", "Pick_and_Place", "Press_Brake",
    "Pump", "Robot_Arm", "Shrink_Wrapper", "Shuttle_System", "Vacuum_Packer",
    "Valve_Controller", "Vision_System", "XRay_Inspector", "3D_Printer"
])

st.subheader("Sensor Readings")
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", min_value=0.0)
    vibration = st.number_input("Vibration (mm/s)", min_value=0.0)
    sound = st.number_input("Sound (dB)", min_value=0.0)
    oil_level = st.number_input("Oil Level (%)", min_value=0.0, max_value=100.0)
    coolant_level = st.number_input("Coolant Level (%)", min_value=0.0, max_value=100.0)
    coolant_flow = st.number_input("Coolant Flow (L/min)", min_value=0.0)
    heat_index = st.number_input("Heat Index", min_value=0.0)

with col2:
    power = st.number_input("Power Consumption (kW)", min_value=0.0)
    last_maintenance = st.number_input("Last Maintenance (Days Ago)", min_value=0)
    maintenance_count = st.number_input("Maintenance History Count", min_value=0)
    failure_count = st.number_input("Failure History Count", min_value=0)
    error_codes = st.number_input("Error Codes Last 30 Days", min_value=0)
    hydraulic = st.number_input("Hydraulic Pressure (bar)", min_value=0.0)
    laser = st.number_input("Laser Intensity", min_value=0.0)

if st.button("Predict..."):

    input_dict = {col: 0 for col in model_columns}

    input_dict['Temperature_C'] = temperature
    input_dict['Vibration_mms'] = vibration
    input_dict['Sound_dB'] = sound
    input_dict['Oil_Level_pct'] = oil_level
    input_dict['Coolant_Level_pct'] = coolant_level
    input_dict['Coolant_Flow_L_min'] = coolant_flow
    input_dict['Heat_Index'] = heat_index
    input_dict['Power_Consumption_kW'] = power
    input_dict['Last_Maintenance_Days_Ago'] = last_maintenance
    input_dict['Maintenance_History_Count'] = maintenance_count
    input_dict['Failure_History_Count'] = failure_count
    input_dict['Error_Codes_Last_30_Days'] = error_codes
    input_dict['Hydraulic_Pressure_bar'] = hydraulic
    input_dict['Laser_Intensity'] = laser

    machine_col = f'Machine_Type_{machine_type}'
    if machine_col in input_dict:
        input_dict[machine_col] = 1

    input_df = pd.DataFrame([input_dict])[model_columns]

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    prediction = 1 if probability >= 0.15 else 0

    st.divider()
    if prediction == 1:
        st.error(f"WARNING!!! This machine is likely to fail within 7 days ({probability*100:.2f}%)")
    else:
        st.success(f"Machine is healthy, no maintenance needed. ({probability*100:.1f}% probability)")