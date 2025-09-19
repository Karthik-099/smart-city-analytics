# Smart City Analytics Platform

## Overview
A data science solution for optimizing city traffic using IoT sensor data and machine learning.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download dataset from [Kaggle](https://www.kaggle.com/datasets/ziya07/smart-mobility-traffic-dataset) to `data/raw/traffic_data.csv`
3. Initialize DVC: `dvc init`
4. Run pipeline: `dvc repro`
5. Start dashboard: `streamlit run deployment/app.py`
6. Run API locally: `python deployment/api.py`
7. View MLflow experiments: `mlflow ui`

## Directory Structure
- `data/`: Raw and processed data
- `notebooks/`: EDA and modeling notebooks
- `src/`: Source code for data processing, modeling, prediction
- `deployment/`: API, dashboard, Docker setup
- `models/`: Trained models
- `reports/`: Business report and model comparison

## Dataset
- Source: Kaggle Smart Mobility Traffic Dataset
- Columns: Date, Time, VehicleCount, AverageSpeed, Occupancy, Latitude, Longitude, etc.

## Deliverables
- EDA notebook: `notebooks/eda.ipynb`
- Trained models: `models/lstm_model.pth`, `models/rf_model.pkl`
- API: Deployed on AWS Lambda or run locally
- Dashboard: Run via Streamlit
- Business report: `reports/business_report.md`