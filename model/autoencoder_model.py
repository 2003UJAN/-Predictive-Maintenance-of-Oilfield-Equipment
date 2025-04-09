import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

def detect_anomalies_autoencoder(df):
    features = ["vibration", "temperature", "pressure"]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[features])

    input_layer = Input(shape=(3,))
    encoded = Dense(2, activation='relu')(input_layer)
    decoded = Dense(3, activation='linear')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(0.001), loss='mse')

    autoencoder.fit(X_scaled, X_scaled, epochs=30, batch_size=32, verbose=0)
    preds = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - preds, 2), axis=1)
    threshold = np.percentile(mse, 95)
    df["anomaly"] = mse > threshold
    return df