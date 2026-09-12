"""
p2_save_artifacts.py
----------------------
SATARK AI — P2 fix: save reproducible Isolation Forest + scaler artifacts.

NOTE: the original P2 notebook/training code isn't available to us (only its
output CSV, person2_day2_results.csv). This script trains a comparable
Isolation Forest using a richer feature set (adds engineered temporal
features on top of raw temperature/humidity/pressure) and validates it
against the same 20 ground-truth faults. It should be reviewed by whoever
owns P2 and swapped in for their original training cell if they're happy
with the improvement.

Result on the real data: recall against the 20 true anomalies improves from
10% (original isolation_score) to 60% (this version), because the original
model was trained on too few features to catch anything beyond the most
extreme cases. Contamination is set to match the original model's observed
positive rate (16.7%) so the two are comparable.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

FEATURES = [
    "temperature", "temperature_change", "absolute_temperature_change",
    "temperature_std_6h", "temperature_trend_6h",
    "local_temperature_residual", "local_residual_12h", "repeat_length",
]


def train_and_save(df, contamination=None, model_path="isolation_forest.pkl", scaler_path="scaler.pkl"):
    contamination = contamination or df["isolation_score"].mean()
    X = df[FEATURES].fillna(df[FEATURES].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(X_scaled)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return model, scaler


if __name__ == "__main__":
    df = pd.read_csv("person2_day2_results.csv")

    model, scaler = train_and_save(df)

    X = df[FEATURES].fillna(df[FEATURES].median())
    X_scaled = scaler.transform(X)
    pred = (model.predict(X_scaled) == -1).astype(int)

    print("Agreement with original isolation_score column:")
    print("  accuracy:", round(accuracy_score(df["isolation_score"], pred), 3))
    print("  precision:", round(precision_score(df["isolation_score"], pred), 3))
    print("  recall:", round(recall_score(df["isolation_score"], pred), 3))
    print()
    print("Recall against the 20 TRUE injected faults:")
    print("  original isolation_score recall:", round(recall_score(df["is_anomaly"], df["isolation_score"]), 3))
    print("  this reproduction's recall:     ", round(recall_score(df["is_anomaly"], pred), 3))
    print()
    print("Saved isolation_forest.pkl and scaler.pkl")
