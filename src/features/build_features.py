import pandas as pd
from typing import Any
import logging

logging.basicConfig(level=logging.INFO)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features."""
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