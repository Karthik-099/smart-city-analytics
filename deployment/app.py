import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os

# Set page configuration
st.set_page_config(page_title="Smart City Traffic Analytics", layout="wide")

# Title
st.title("Smart City Traffic Analytics Dashboard")

# Load data
data_path = os.path.join("data", "processed", "features.csv")
df = pd.read_csv(data_path, parse_dates=['Timestamp'])

# Calculate KPIs
congestion_rate = df['Congestion'].mean() * 100
avg_vehicle_count = df['VehicleCount'].mean()
avg_speed = df['AverageSpeed'].mean()

# Display KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Congestion Rate", f"{congestion_rate:.2f}%")
col2.metric("Average Vehicle Count", f"{avg_vehicle_count:.2f}")
col3.metric("Average Speed", f"{avg_speed:.2f} km/h")

# Traffic Trends Over Time
st.subheader("Traffic Trends Over Time")
fig = px.line(df, x='Timestamp', y='VehicleCount', title='Vehicle Count Over Time')
st.plotly_chart(fig, use_container_width=True)

# Congestion Prediction
st.subheader("Congestion Prediction")
with st.form("prediction_form"):
    hour = st.slider("Hour of Day", 0, 23, 8)
    day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 1)
    is_weekend = st.selectbox("Is Weekend?", [0, 1], index=0)
    rush_hour = st.selectbox("Rush Hour?", [0, 1], index=1)
    vehicle_count_lag1 = st.number_input("Previous Hour Vehicle Count", value=150.0)
    avg_speed_lag1 = st.number_input("Previous Hour Average Speed", value=30.0)
    vehicle_count_rolling_mean = st.number_input("Rolling Mean Vehicle Count", value=145.0)
    submitted = st.form_submit_button("Predict")

    if submitted:
        # Load model
        model_path = os.path.join("models", "rf_model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Prepare input
        input_data = pd.DataFrame({
            'Hour': [hour],
            'DayOfWeek': [day_of_week],
            'IsWeekend': [is_weekend],
            'RushHour': [rush_hour],
            'VehicleCount_lag1': [vehicle_count_lag1],
            'AverageSpeed_lag1': [avg_speed_lag1],
            'VehicleCount_rolling_mean': [vehicle_count_rolling_mean]
        })

        # Predict
        prediction = model.predict(input_data)[0]
        st.write(f"Congestion: {'Yes' if prediction == 1 else 'No'}")