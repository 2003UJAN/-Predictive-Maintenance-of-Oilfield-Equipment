import streamlit as st
import pandas as pd
import plotly.express as px
from utils.helpers import load_data
from model.autoencoder_model import detect_anomalies_autoencoder
from model.isolation_forest_model import detect_anomalies_isolation_forest

# Set Streamlit page config
st.set_page_config(page_title="Oilfield Predictive Maintenance", layout="wide")

# Online oilfield-themed image (Unsplash)
st.image("https://images.unsplash.com/photo-1579389083078-631d8d4e8b67?auto=format&fit=crop&w=1350&q=80", use_column_width=True)

st.title("🛢️ Predictive Maintenance for Oilfield Equipment")
st.markdown("Detect anomalies in pumps, compressors, and motors using AI.")

# Load data
df = load_data("data/synthetic_sensor_data.csv")

# Sidebar model selector
st.sidebar.header("Choose Model")
model_choice = st.sidebar.selectbox("Anomaly Detection Model", ["Autoencoder", "Isolation Forest"])

# Apply model
if model_choice == "Autoencoder":
    df_anom = detect_anomalies_autoencoder(df)
else:
    df_anom = detect_anomalies_isolation_forest(df)

# Display data
if st.checkbox("Show Raw Data"):
    st.dataframe(df_anom.head())

# Plot vibration anomalies
fig = px.scatter(df_anom, x="time", y="vibration", color="anomaly",
                 title="Vibration Anomaly Detection", labels={"time": "Timestamp"})
st.plotly_chart(fig, use_container_width=True)

# Status
num_anomalies = (df_anom["anomaly"] == "Anomaly").sum()
st.success(f"🔍 Detected **{num_anomalies} anomalies** out of **{len(df_anom)} records**.")
