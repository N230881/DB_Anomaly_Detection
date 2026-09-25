"""
LSTM Time-Series Anomaly Detection — Database Failure Prediction
==================================================================
This script:
1. Downloads REAL AWS server CPU usage data (with confirmed real anomalies)
2. Trains an LSTM Autoencoder on the "normal" (healthy) portion of the data
3. Tests it on the full timeline, including the real anomaly period
4. Saves a results.csv and a plot showing predicted vs real anomalies

Run this file first: python train_model.py
Then run the dashboard: streamlit run app.py
"""

import os
import json
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so it works without a display window
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

# ---------------------------------------------------------------------------
# STEP 1: Make sure our folders exist
# ---------------------------------------------------------------------------
FOLDERS = [
    "data/realAWSCloudwatch",
    "labels",
    "model",
    "outputs",
]
for f in FOLDERS:
    os.makedirs(f, exist_ok=True)

DATA_PATH = "data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"
LABELS_PATH = "labels/combined_labels.json"
LABEL_KEY = "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"

# ---------------------------------------------------------------------------
# STEP 2: Download the REAL data + REAL official anomaly labels (only if missing)
# ---------------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    print("Downloading real AWS server data...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
        DATA_PATH,
    )

if not os.path.exists(LABELS_PATH):
    print("Downloading real confirmed anomaly labels...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_labels.json",
        LABELS_PATH,
    )

# ---------------------------------------------------------------------------
# STEP 3: Load data
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

with open(LABELS_PATH) as f:
    labels = json.load(f)

real_anomaly_times = pd.to_datetime(labels[LABEL_KEY])
df["is_real_anomaly"] = df["timestamp"].isin(real_anomaly_times)

print(f"Loaded {len(df)} rows.")
print(f"Confirmed real anomaly timestamps: {list(real_anomaly_times)}")

# ---------------------------------------------------------------------------
# STEP 4: Scale + create sliding windows (sequences)
# ---------------------------------------------------------------------------
WINDOW_SIZE = 30  # last 30 readings (2.5 hours, since data is every 5 minutes)

scaler = MinMaxScaler()
df["scaled_value"] = scaler.fit_transform(df[["value"]])


def create_sequences(data, window):
    sequences = []
    for i in range(len(data) - window):
        sequences.append(data[i:i + window])
    return np.array(sequences)


X = create_sequences(df["scaled_value"].values.reshape(-1, 1), WINDOW_SIZE)
print("Prepared sequence shape:", X.shape)

# ---------------------------------------------------------------------------
# STEP 5: Split into train (normal-only, before the real anomaly) and test (all)
# ---------------------------------------------------------------------------
anomaly_start_index = df[df["is_real_anomaly"]].index.min()
train_cutoff = max(anomaly_start_index - WINDOW_SIZE - 50, 50)

X_train = X[:train_cutoff]
X_test = X

print(f"Training on {X_train.shape[0]} normal samples.")
print(f"Testing on {X_test.shape[0]} samples (includes the real anomaly).")

# ---------------------------------------------------------------------------
# STEP 6: Build the LSTM Autoencoder
# ---------------------------------------------------------------------------
model = Sequential([
    LSTM(32, activation="relu", input_shape=(WINDOW_SIZE, 1), return_sequences=False),
    RepeatVector(WINDOW_SIZE),
    LSTM(32, activation="relu", return_sequences=True),
    TimeDistributed(Dense(1)),
])
model.compile(optimizer="adam", loss="mse")
model.summary()

# ---------------------------------------------------------------------------
# STEP 7: Train
# ---------------------------------------------------------------------------
print("Training model... (this takes ~1-2 minutes on a normal laptop)")
model.fit(X_train, X_train, epochs=25, batch_size=32, validation_split=0.1, verbose=1)

model.save("model/lstm_anomaly_model.h5")
print("Model saved to model/lstm_anomaly_model.h5")

# ---------------------------------------------------------------------------
# STEP 8: Detect anomalies using reconstruction error
# ---------------------------------------------------------------------------
X_pred = model.predict(X_test, verbose=0)
reconstruction_error = np.mean(np.abs(X_pred - X_test), axis=(1, 2))

threshold = np.percentile(reconstruction_error[:train_cutoff], 95)
predicted_anomaly = reconstruction_error > threshold

print(f"Anomaly threshold: {threshold:.5f}")
print(f"Points flagged as anomaly by the AI: {predicted_anomaly.sum()}")

# ---------------------------------------------------------------------------
# STEP 9: Save results + plot
# ---------------------------------------------------------------------------
result_df = df.iloc[WINDOW_SIZE:].copy()
result_df["reconstruction_error"] = reconstruction_error
result_df["threshold"] = threshold
result_df["predicted_anomaly"] = predicted_anomaly

result_df.to_csv("outputs/results.csv", index=False)
print("Results saved to outputs/results.csv")

plt.figure(figsize=(14, 5))
plt.plot(result_df["timestamp"], result_df["reconstruction_error"], label="AI Error Score")
plt.axhline(threshold, color="red", linestyle="--", label="Anomaly Threshold")
real_points = result_df[result_df["is_real_anomaly"]]
plt.scatter(real_points["timestamp"], real_points["reconstruction_error"],
            color="green", marker="x", s=100, label="REAL Confirmed Anomaly")
plt.legend()
plt.title("AI Detection vs Real Confirmed Anomaly")
plt.xlabel("Time")
plt.ylabel("Reconstruction Error")
plt.tight_layout()
plt.savefig("outputs/anomaly_plot.png")
print("Plot saved to outputs/anomaly_plot.png")

print("\nDONE. Now run:  streamlit run app.py")
