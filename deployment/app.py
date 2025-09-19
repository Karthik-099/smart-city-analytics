import streamlit as st
import pandas as pd
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Define LSTM model class (must match train_model.py)
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        _, (h, _) = self.lstm(x.unsqueeze(1))
        out = self.fc(h[-1])
        return self.sigmoid(out)

# Load data
df = pd.read_csv('/home/karthik/smart-city-analytics/data/processed/features.csv', parse_dates=['Timestamp'])

# Load models
with open('/home/karthik/smart-city-analytics/models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
lstm_model = LSTMModel()
lstm_model.load_state_dict(torch.load('/home/karthik/smart-city-analytics/models/lstm_model.pth'))
lstm_model.eval()

# Fit scaler
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
model_choice = st.selectbox("Select Model", ["Random Forest", "LSTM"], index=0)
hour = st.number_input("Hour", min_value=0, max_value=23, value=8)
day_of_week = st.number_input("Day of Week", min_value=0, max_value=6, value=1)
is_weekend = st.selectbox("Is Weekend", [0, 1], index=0)
rush_hour = st.selectbox("Rush Hour", [0, 1], index=1)
vehicle_count_lag1 = st.number_input("Vehicle Count Lag1", value=150.0)
speed_lag1 = st.number_input("Speed Lag1", value=30.0)
vehicle_rolling_mean = st.number_input("Vehicle Rolling Mean", value=145.0)

if st.button("Predict"):
    input_df = pd.DataFrame({
        'Hour': [hour],
        'DayOfWeek': [day_of_week],
        'IsWeekend': [is_weekend],
        'RushHour': [rush_hour],
        'VehicleCount_lag1': [vehicle_count_lag1],
        'AverageSpeed_lag1': [speed_lag1],
        'VehicleCount_rolling_mean': [vehicle_rolling_mean]
    })
    input_scaled = scaler.transform(input_df)
    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_scaled)[0]
        st.write(f"Congestion (Random Forest): {'Yes' if prediction == 1 else 'No'}")
    else:
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        with torch.no_grad():
            prediction = lstm_model(input_tensor).round().item()
        st.write(f"Congestion (LSTM): {'Yes' if prediction == 1 else 'No'}")