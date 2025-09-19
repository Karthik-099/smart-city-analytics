import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

st.title("Smart City Traffic Dashboard")

df = pd.read_csv('data/processed/features.csv', parse_dates=['Timestamp'])

st.subheader("Key Performance Indicators")
st.metric("Average Speed", df['AverageSpeed'].mean())
st.metric("Average Vehicle Count", df['VehicleCount'].mean())
st.metric("Congestion Rate", f"{df['Congestion'].mean() * 100:.2f}%")

st.subheader("Traffic Trends")
fig, ax = plt.subplots()
ax.plot(df['Timestamp'], df['VehicleCount'])
st.pyplot(fig)

st.subheader("Predict Congestion")
with st.form(key='predict_form'):
    hour = st.number_input('Hour', 0, 23)
    dayofweek = st.number_input('Day of Week', 0, 6)
    isweekend = st.selectbox('Is Weekend', [0, 1])
    rushhour = st.selectbox('Rush Hour', [0, 1])
    vclag = st.number_input('Vehicle Count Lag1')
    speedlag = st.number_input('Speed Lag1')
    vcrolling = st.number_input('Vehicle Rolling Mean')
    submit = st.form_submit_button('Predict')

if submit:
    data = [{'Hour': hour, 'DayOfWeek': dayofweek, 'IsWeekend': isweekend, 'RushHour': rushhour,
             'VehicleCount_lag1': vclag, 'AverageSpeed_lag1': speedlag, 'VehicleCount_rolling_mean': vcrolling}]
    response = requests.post('http://localhost:5000/predict', json=data)
    if response.ok:
        pred = response.json()['congestion'][0]
        st.write(f"Predicted Congestion: {'Yes' if pred == 1 else 'No'}")
    else:
        st.error("API error")