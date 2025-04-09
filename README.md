# 🛢️ Predictive Maintenance of Oilfield Equipment

A Streamlit-based interactive dashboard for detecting failures in pumps, compressors, and drilling motors using sensor time-series data and anomaly detection models such as Autoencoders and Isolation Forest.

![Oilfield Banner](assets/oilfield_banner.png)

---

## 📌 Project Overview

This project focuses on **Predictive Maintenance** for critical oilfield equipment by:
- Simulating realistic **sensor data** (vibration, temperature, pressure).
- Detecting potential failures using **Autoencoders** (deep learning) and **Isolation Forests** (unsupervised learning).
- Visualizing anomalies via a sleek **Streamlit UI** inspired by petroleum reservoir graphics.

---

## 🧠 Features

✅ Generate synthetic sensor data  
✅ Train and evaluate anomaly detection models  
✅ Interactive visualization of detected anomalies  
✅ Cool petroleum-themed user interface (UI)  
✅ Minimal setup and lightweight dashboard  

---

## 🗂️ Repository Structure

predictive-maintenance-oilfield/ │ ├── app.py # Streamlit dashboard with anomaly detection ├── generate_data.py # Sensor data generator (synthetic) ├── requirements.txt # All dependencies │ ├── model/ │ ├── autoencoder_model.py # Autoencoder model definition │ └── isolation_forest_model.py# Isolation Forest implementation │ ├── utils/ │ └── helpers.py # Helper functions for plots, preprocessing │ ├── data/ │ ├── synthetic_sensor_data.csv # Generated time-series dataset │ ├── assets/ │ └── oilfield_banner.png # UI banner image │ └── README.md # Project documentation


---

## 🚀 Quick Start

### 1️⃣ Clone this repository
```bash
git clone https://github.com/your-username/predictive-maintenance-oilfield.git
cd predictive-maintenance-oilfield

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Generate synthetic sensor data

python generate_data.py

4️⃣ Launch the Streamlit dashboard

streamlit run app.py

🧪 Models Used
Model	Description
🔹 Autoencoder	Unsupervised deep learning model trained to reconstruct input data. High reconstruction error = anomaly.
🔹 Isolation Forest	Tree-based anomaly detection algorithm that isolates outliers via recursive partitioning.
