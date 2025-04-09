import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies_isoforest(df):
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    features = ["vibration", "temperature", "pressure"]
    df["anomaly"] = model.fit_predict(df[features])
    df["anomaly"] = df["anomaly"] == -1
    return df