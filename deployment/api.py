from flask import Flask, request, jsonify
import torch
import pandas as pd
from src.models.train_model import LSTMModel

app = Flask(__name__)

input_size = 7
model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2)
model.load_state_dict(torch.load('models/lstm_model.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    features = ['Hour', 'DayOfWeek', 'IsWeekend', 'RushHour', 'VehicleCount_lag1', 'AverageSpeed_lag1', 'VehicleCount_rolling_mean']
    X = torch.tensor(df[features].values, dtype=torch.float32)
    with torch.no_grad():
        pred = model(X).round().numpy()
    return jsonify({'congestion': pred.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)