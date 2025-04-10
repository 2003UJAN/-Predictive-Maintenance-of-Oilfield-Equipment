import pandas as pd
import numpy as np
import os

# Create synthetic sensor data for vibration, pressure, and temperature
def generate_sensor_data(num_samples=1000):
    time = pd.date_range(start="2023-01-01", periods=num_samples, freq="H")
    vibration = np.random.normal(loc=0.5, scale=0.05, size=num_samples)
    pressure = np.random.normal(loc=30, scale=3, size=num_samples)
    temperature = np.random.normal(loc=70, scale=5, size=num_samples)

    # Inject anomalies
    for i in np.random.choice(num_samples, size=int(0.05*num_samples), replace=False):
        vibration[i] += np.random.uniform(0.2, 0.5)
        pressure[i] += np.random.uniform(10, 20)
        temperature[i] += np.random.uniform(10, 15)

    df = pd.DataFrame({
        "timestamp": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature
    })

    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_sensor_data()
    df.to_csv("data/synthetic_sensor_data.csv", index=False)
    print("✅ Synthetic sensor data generated and saved to data/synthetic_sensor_data.csv")
