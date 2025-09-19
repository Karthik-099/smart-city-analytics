import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('/home/karthik/smart-city-analytics/data/processed/features.csv', parse_dates=['Timestamp'])

# Load Random Forest model
with open('/home/karthik/smart-city-analytics/models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Fit scaler on training features
features = ['Hour', 'DayOfWeek', 'IsWeekend', 'RushHour', 'VehicleCount_lag1', 'AverageSpeed_lag1', 'VehicleCount_rolling_mean']
scaler = StandardScaler()
scaler.fit(df[features])

# KPIs
st.title("Smart City Traffic Dashboard")
st.header("Key Performance Indicators")
st.metric("Average Speed", f"{df['AverageSpeed'].mean():.2f} km/h")
st.metric("Average Vehicle Count", f"{df['VehicleCount'].mean():.2f}")
st.metric("Congestion Rate", f"{(df['Congestion'].mean() * 100):.2f}%")

# Traffic Trends Chart
st.header("Traffic Trends Chart")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['Timestamp'], df['VehicleCount'], label='Vehicle Count', alpha=0.7)
ax.plot(df['Timestamp'], df['AverageSpeed'], label='Average Speed (km/h)', alpha=0.7)
ax.set_title('Traffic Metrics Over Time')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Prediction Form
st.header("Predict Congestion")
hour = st.number_input("Hour", min_value=0, max_value=23, value=8)
day_of_week = st.number_input("Day of Week", min_value=0, max_value=6, value=1)
is_weekend = st.selectbox("Is Weekend", [0, 1], index=0)
rush_hour = st.selectbox("Rush Hour", [0, 1], index=1)
vehicle_count_lag1 = st.number_input("Vehicle Count Lag1", value=150.0)
speed_lag1 = st.number_input("Speed Lag1", value=30.0)
vehicle_rolling_mean = st.number_input("Vehicle Rolling Mean", value=145.0)

if st.button("Predict"):
    # Create DataFrame with feature names
    input_df = pd.DataFrame({
        'Hour': [hour],
        'DayOfWeek': [day_of_week],
        'IsWeekend': [is_weekend],
        'RushHour': [rush_hour],
        'VehicleCount_lag1': [vehicle_count_lag1],
        'AverageSpeed_lag1': [speed_lag1],
        'VehicleCount_rolling_mean': [vehicle_rolling_mean]
    })
    # Scale inputs
    input_scaled = scaler.transform(input_df)
    # Predict with Random Forest
    prediction = rf_model.predict(input_scaled)[0]
    st.write(f"Congestion: {'Yes' if prediction == 1 else 'No'}")