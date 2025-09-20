import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, request, jsonify
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.models.train_model import LSTMModel

app = Flask(__name__)


df = pd.read_csv('/home/karthik/smart-city-analytics/data/processed/features.csv')
features = ['Hour', 'DayOfWeek', 'IsWeekend', 'RushHour', 'VehicleCount_lag1', 'AverageSpeed_lag1', 'VehicleCount_rolling_mean']
scaler = StandardScaler()
scaler.fit(df[features])


input_size = 7
model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2)
model.load_state_dict(torch.load('/home/karthik/smart-city-analytics/models/lstm_model.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data], columns=features)
        X_scaled = scaler.transform(df[features])
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            pred = model(X_tensor).round().numpy()
        return jsonify({'congestion': int(pred[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)