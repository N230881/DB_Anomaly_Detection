"""
Streamlit Dashboard — LSTM Database Anomaly Detection
=======================================================
Run this AFTER train_model.py has been run once (it needs outputs/results.csv).

To run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="DB Anomaly Detection", layout="wide")

st.title("🖥️ Database Health Monitor — LSTM Anomaly Detection")
st.write(
    "This dashboard shows a real AWS server's CPU usage over time, "
    "and highlights where the AI model detected unusual behavior that "
    "could signal an upcoming failure."
)

RESULTS_PATH = "outputs/results.csv"

if not os.path.exists(RESULTS_PATH):
    st.error(
        "No results found yet. Please run `python train_model.py` first "
        "to train the model and generate outputs/results.csv."
    )
    st.stop()

df = pd.read_csv(RESULTS_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# --- Top metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Total data points", len(df))
col2.metric("AI-flagged anomaly points", int(df["predicted_anomaly"].sum()))
col3.metric("Real confirmed anomaly points", int(df["is_real_anomaly"].sum()))

st.divider()

# --- Raw CPU usage chart ---
st.subheader("📈 Server CPU Usage Over Time")
fig1, ax1 = plt.subplots(figsize=(14, 4))
ax1.plot(df["timestamp"], df["value"], label="CPU Usage", color="steelblue")
real_points = df[df["is_real_anomaly"]]
ax1.scatter(real_points["timestamp"], real_points["value"],
            color="green", marker="x", s=100, label="Real Confirmed Anomaly", zorder=5)
ax1.set_xlabel("Time")
ax1.set_ylabel("CPU %")
ax1.legend()
st.pyplot(fig1)

# --- Anomaly score chart ---
st.subheader("🚨 AI Anomaly Detection Score")
fig2, ax2 = plt.subplots(figsize=(14, 4))
ax2.plot(df["timestamp"], df["reconstruction_error"], label="AI Error Score", color="darkorange")
ax2.axhline(df["threshold"].iloc[0], color="red", linestyle="--", label="Anomaly Threshold")
ax2.scatter(real_points["timestamp"], real_points["reconstruction_error"],
            color="green", marker="x", s=100, label="Real Confirmed Anomaly", zorder=5)
ax2.set_xlabel("Time")
ax2.set_ylabel("Reconstruction Error")
ax2.legend()
st.pyplot(fig2)

st.divider()

# --- Flagged anomaly table ---
st.subheader("🔍 Points the AI flagged as risky")
flagged = df[df["predicted_anomaly"]][["timestamp", "value", "reconstruction_error"]]
st.dataframe(flagged, use_container_width=True)

st.caption(
    "Green X marks = officially confirmed real anomaly (from AWS incident records). "
    "Orange line = how 'unusual' the AI thinks each moment is. "
    "When the orange line crosses the red dashed threshold, the AI is raising a warning."
)
