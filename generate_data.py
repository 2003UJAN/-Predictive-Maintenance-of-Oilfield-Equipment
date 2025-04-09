import pandas as pd
import numpy as np

def generate_sensor_data(num_samples=1000):
    np.random.seed(42)
    time = np.arange(num_samples)
    vibration = np.sin(0.02 * time) + np.random.normal(0, 0.1, size=num_samples)
    temperature = 60 + 5 * np.sin(0.01 * time) + np.random.normal(0, 0.5, size=num_samples)
    pressure = 30 + 2 * np.cos(0.015 * time) + np.random.normal(0, 0.3, size=num_samples)
    
    anomalies = np.random.choice(num_samples, size=int(0.05 * num_samples), replace=False)
    vibration[anomalies] += np.random.normal(1, 0.5, size=anomalies.shape[0])
    
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "temperature": temperature,
        "pressure": pressure
    })
    
    df.to_csv("data/synthetic_sensor_data.csv", index=False)
    print("Synthetic sensor data saved to 'data/synthetic_sensor_data.csv'.")

if __name__ == "__main__":
    generate_sensor_data()