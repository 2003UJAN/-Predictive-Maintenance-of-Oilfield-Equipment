import streamlit as st
import pandas as pd
import plotly.express as px
from model.autoencoder_model import detect_anomalies_autoencoder
from model.isolation_forest_model import detect_anomalies_isoforest
from utils.helpers import load_data

st.set_page_config(page_title="Oilfield Predictive Maintenance", layout="wide")

st.image("assets/oilfield_banner.png", use_column_width=True)
st.title("🛢️ Oilfield Predictive Maintenance Dashboard")

data_option = st.sidebar.selectbox("Select Data", ["Synthetic Data"])
model_option = st.sidebar.radio("Anomaly Detection Model", ["Autoencoder", "Isolation Forest"])

df = load_data("data/synthetic_sensor_data.csv")
st.write("### Sensor Data Overview", df.head())

if model_option == "Autoencoder":
    df_anom = detect_anomalies_autoencoder(df)
else:
    df_anom = detect_anomalies_isoforest(df)

st.write("### Detected Anomalies")
fig = px.scatter(df_anom, x="time", y="vibration", color="anomaly", title="Vibration Anomaly Detection")
st.plotly_chart(fig, use_container_width=True)