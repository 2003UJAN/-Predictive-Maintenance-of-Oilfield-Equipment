import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies_isolation_forest(df):
    df = df.copy()
    df['time'] = df['timestamp']
    model = IsolationForest(contamination=0.05, random_state=42)
    features = ['vibration', 'pressure', 'temperature']
    df['anomaly'] = model.fit_predict(df[features])
    df['anomaly'] = df['anomaly'].map({1: "Normal", -1: "Anomaly"})
    return df
