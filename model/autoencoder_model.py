import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def detect_anomalies_autoencoder(df):
    df = df.copy()
    df['time'] = df['timestamp']
    features = ['vibration', 'pressure', 'temperature']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Autoencoder architecture
    model = Sequential([
        Dense(16, activation='relu', input_shape=(3,)),
        Dense(8, activation='relu'),
        Dense(3, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_scaled, X_scaled, epochs=10, batch_size=32, verbose=0)

    # Predict and calculate reconstruction error
    X_pred = model.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
    threshold = np.percentile(mse, 95)

    df['anomaly'] = ['Anomaly' if e > threshold else 'Normal' for e in mse]
    return df
