import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import pickle
import yaml
import logging

logging.basicConfig(level=logging.INFO)

class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h, _) = self.lstm(x.unsqueeze(1))
        out = self.fc(h[-1])
        return self.sigmoid(out)

def train_lstm(X_train, y_train, X_test, y_test, params):
    train_dataset = TrafficDataset(X_train, y_train)
    test_dataset = TrafficDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=params['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['training']['batch_size'])

    model = LSTMModel(input_size=X_train.shape[1], hidden_size=params['model']['lstm_hidden_size'], num_layers=params['model']['lstm_num_layers'])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with mlflow.start_run(run_name="LSTM"):
        mlflow.log_params(params['model'])
        for epoch in range(params['training']['epochs']):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                y_pred.extend(outputs.round().cpu().numpy())
                y_true.extend(y_batch.cpu().numpy())
        acc = accuracy_score(y_true, y_pred)
        mlflow.log_metric("accuracy", acc)
        mlflow.pytorch.log_model(model, "lstm_model")
        torch.save(model.state_dict(), 'models/lstm_model.pth')
    return model, acc

def train_rf(X_train, y_train, X_test, y_test, params):
    model = RandomForestClassifier(n_estimators=params['model']['rf_n_estimators'], random_state=42)
    with mlflow.start_run(run_name="RandomForest"):
        mlflow.log_params(params['model'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "rf_model")
        with open('models/rf_model.pkl', 'wb') as f:
            pickle.dump(model, f)
    return model, acc

if __name__ == "__main__":
    with open('params.yaml') as f:
        params = yaml.safe_load(f)
    df = pd.read_csv('data/processed/features.csv', parse_dates=['Timestamp'])
    features = ['Hour', 'DayOfWeek', 'IsWeekend', 'RushHour', 'VehicleCount_lag1', 'AverageSpeed_lag1', 'VehicleCount_rolling_mean']
    X = df[features]
    y = df['Congestion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['training']['test_size'], shuffle=False)
    lstm_model, lstm_acc = train_lstm(X_train, y_train, X_test, y_test, params)
    rf_model, rf_acc = train_rf(X_train, y_train, X_test, y_test, params)
    logging.info(f"LSTM Accuracy: {lstm_acc}, RF Accuracy: {rf_acc}")

    report = f"LSTM Acc: {lstm_acc}
RF Acc: {rf_acc}
Better model: {'LSTM' if lstm_acc > rf_acc else 'RF'}"
    with open('reports/model_comparison.txt', 'w') as f:
        f.write(report)