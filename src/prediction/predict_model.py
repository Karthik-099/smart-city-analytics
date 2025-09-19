import torch
import pandas as pd
from src.models.train_model import LSTMModel

def predict_congestion(data: dict, model_path: str = 'models/lstm_model.pth') -> int:
    """Predict congestion from input data."""
    model = LSTMModel(input_size=7, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    features = ['Hour', 'DayOfWeek', 'IsWeekend', 'RushHour', 'VehicleCount_lag1', 'AverageSpeed_lag1', 'VehicleCount_rolling_mean']
    df = pd.DataFrame([data])[features]
    X = torch.tensor(df.values, dtype=torch.float32)
    with torch.no_grad():
        pred = model(X).round().item()
    return int(pred)