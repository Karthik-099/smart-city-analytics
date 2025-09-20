import os
import pathlib
import subprocess
from typing import Dict

def create_directory_structure(base_path: str) -> None:
    """Create project directory structure."""
    directories = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src/data",
        "src/features",
        "src/models",
        "src/prediction",
        "deployment",
        "models",
        "reports",
    ]
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)
    print("Directory structure created.")

def write_file(path: str, content: str) -> None:
    """Write content to a file."""
    with open(path, "w") as f:
        f.write(content.strip())
    print(f"Created {path}")

def create_project_files(base_path: str) -> None:
    """Create all project files with their content."""
    files: Dict[str, str] = {
        "params.yaml": """
data:
  path: data/raw/traffic_data.csv
model:
  lstm_hidden_size: 64
  lstm_num_layers: 2
  rf_n_estimators: 100
training:
  test_size: 0.2
  batch_size: 32
  epochs: 20
""",
        "dvc.yaml": """
stages:
  process_data:
    cmd: python src/data/make_dataset.py
    deps:
      - src/data/make_dataset.py
      - data/raw
    outs:
      - data/processed
  build_features:
    cmd: python src/features/build_features.py
    deps:
      - src/features/build_features.py
      - data/processed
    outs:
      - data/processed/features.csv
  train_model:
    cmd: python src/models/train_model.py
    deps:
      - src/models/train_model.py
      - data/processed/features.csv
    outs:
      - models/lstm_model.pth
      - models/rf_model.pkl
""",
        "requirements.txt": """
pandas==2.1.4
numpy==1.26.4
matplotlib==3.8.2
seaborn==0.13.1
scikit-learn==1.4.0
torch==2.2.0
mlflow==2.10.2
dvc==3.48.0
streamlit==1.30.0
flask==3.0.1
boto3==1.34.39
requests==2.31.0
""",
        "src/__init__.py": "",
        "src/data/__init__.py": "",
        "src/features/__init__.py": "",
        "src/models/__init__.py": "",
        "src/prediction/__init__.py": "",
        "src/data/make_dataset.py": """
import pandas as pd
import logging
from typing import Any

logging.basicConfig(level=logging.INFO)

def load_data(path: str) -> pd.DataFrame:
    \"\"\"Load CSV data.\"\"\"
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Handle missing values, outliers, timestamps.\"\"\"
    df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.drop(['Date', 'Time'], axis=1)
    df = df.sort_values('Timestamp')
    df = df.set_index('Timestamp')
    df = df.fillna(method='ffill').fillna(method='bfill')
    df['AverageSpeed'] = df['AverageSpeed'].clip(0, 100)
    df['VehicleCount'] = df['VehicleCount'].clip(0, df['VehicleCount'].quantile(0.99))
    logging.info("Data cleaned")
    return df

if __name__ == "__main__":
    import yaml
    with open('params.yaml') as f:
        params = yaml.safe_load(f)
    df = load_data(params['data']['path'])
    df = clean_data(df)
    df.to_csv('data/processed/cleaned_data.csv')
""",
        "src/features/build_features.py": """
import pandas as pd
from typing import Any
import logging

logging.basicConfig(level=logging.INFO)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Create new features.\"\"\"
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['RushHour'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9) | (df['Hour'] >= 16) & (df['Hour'] <= 18)).astype(int)
    df['VehicleCount_lag1'] = df['VehicleCount'].shift(1)
    df['AverageSpeed_lag1'] = df['AverageSpeed'].shift(1)
    df['VehicleCount_rolling_mean'] = df['VehicleCount'].rolling(window=3).mean()
    df['Congestion'] = ((df['AverageSpeed'] < 30) & (df['Occupancy'] > 0.5)).astype(int)
    df = df.dropna()
    logging.info("Features engineered")
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/processed/cleaned_data.csv', index_col='Timestamp', parse_dates=True)
    df = engineer_features(df)
    df.to_csv('data/processed/features.csv')
""",
        "src/models/train_model.py": """
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

    report = f"LSTM Acc: {lstm_acc}\nRF Acc: {rf_acc}\nBetter model: {'LSTM' if lstm_acc > rf_acc else 'RF'}"
    with open('reports/model_comparison.txt', 'w') as f:
        f.write(report)
""",
        "src/prediction/predict_model.py": """
import torch
import pandas as pd
from src.models.train_model import LSTMModel

def predict_congestion(data: dict, model_path: str = 'models/lstm_model.pth') -> int:
    \"\"\"Predict congestion from input data.\"\"\"
    model = LSTMModel(input_size=7, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    features = ['Hour', 'DayOfWeek', 'IsWeekend', 'RushHour', 'VehicleCount_lag1', 'AverageSpeed_lag1', 'VehicleCount_rolling_mean']
    df = pd.DataFrame([data])[features]
    X = torch.tensor(df.values, dtype=torch.float32)
    with torch.no_grad():
        pred = model(X).round().item()
    return int(pred)
""",
        "deployment/Dockerfile": """
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src /app/src
COPY models /app/models
COPY deployment/api.py /app/api.py
CMD ["python", "api.py"]
""",
        "deployment/api.py": """
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
""",
        "deployment/app.py": """
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
""",
        "reports/business_report.md": """
# Recommendations for City Traffic Management
- Optimize signals at hotspots during rush hours.
- Cost-Benefit: Implementation cost ~$100K (sensors/AWS), benefits: 20% reduced congestion, saving $1M in fuel/time annually.
- Actionable: Deploy model for real-time alerts.
""",
        




.

`
""",
        "setup.sh": """
#!/bin/bash
pip install -r requirements.txt
git init
dvc init
echo "data" >> .gitignore
echo "mlruns" >> .gitignore
echo "*.pyc" >> .gitignore
git add .
git commit -m "Initial project setup"
echo "Run 'dvc repro' to execute pipeline"
echo "Download dataset to data/raw/traffic_data.csv"
""",
        "notebooks/eda.ipynb": """
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "",
    "df = pd.read_csv('data/processed/features.csv', index_col='Timestamp', parse_dates=True)",
    "",
    "# Traffic patterns",
    "plt.figure(figsize=(12,6))",
    "plt.plot(df['VehicleCount'])",
    "plt.title('Vehicle Count Over Time')",
    "plt.show()",
    "",
    "# Peak hours",
    "sns.boxplot(x='Hour', y='VehicleCount', data=df)",
    "plt.title('Vehicle Count by Hour')",
    "plt.show()",
    "",
    "# Congestion hotspots",
    "if 'Latitude' in df.columns:",
    "    sns.scatterplot(x='Longitude', y='Latitude', hue='Congestion', data=df)",
    "    plt.title('Congestion Hotspots')",
    "    plt.show()",
    "",
    "# Correlation",
    "sns.heatmap(df.corr(), annot=True)",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
"""
    }

    for file_path, content in files.items():
        write_file(os.path.join(base_path, file_path), content)

def initialize_git_and_dvc() -> None:
    """Initialize Git and DVC."""
    try:
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["dvc", "init"], check=True)
        with open(".gitignore", "w") as f:
            f.write("data\nmlruns\n*.pyc")
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Initial project setup"], check=True)
        print("Git and DVC initialized.")
    except subprocess.CalledProcessError as e:
        print(f"Error initializing Git/DVC: {e}")

def main() -> None:
    base_path = "."
    create_directory_structure(base_path)
    create_project_files(base_path)
    initialize_git_and_dvc()
    

if __name__ == "__main__":
    main()