"""
anomaly_detector.py
--------------------
SATARK AI — Person 2 deliverable
Isolation Forest + Statistical + Temporal Anomaly Detection Engine

Interface (agreed team contract):
    Input  : DataFrame with at least
             [timestamp, station_id, temperature, humidity, pressure]
    Output : same DataFrame + these new columns
             [statistical_anomaly, isolation_anomaly, temporal_anomaly,
              ml_anomaly_score, temporal_score, satark_score,
              satark_prediction, severity, reason]

Hand this DataFrame to P3 (spatial + diagnosis) and P4 (dashboard).
Do NOT feed is_anomaly / fault_type / original_* columns into the model —
those are ground-truth-only columns used later for evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

CORE_FEATURES = ["temperature", "pressure", "humidity"]
ZSCORE_COLS = [
    "temperature_zscore_24h",
    "pressure_zscore_24h",
    "humidity_zscore_24h",
]

# Columns that must NEVER be used as model features (ground truth / leakage)
LEAKAGE_COLUMNS = [
    "is_anomaly", "fault_type", "fault_severity",
    "original_temperature", "original_pressure", "original_humidity",
]


# ---------------------------------------------------------------------
# 1. Rolling temporal features (skip this step if your df already has
#    the *_rolling_mean_24h / *_zscore_24h columns from processed_meteostat_data.csv)
# ---------------------------------------------------------------------
def create_rolling_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """24h rolling mean/std/deviation/zscore using only PAST data (shift(1))."""
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    rolling_mean = (
        df.groupby("station_id", group_keys=False)
        .apply(lambda x: x.set_index("timestamp")[column]
               .shift(1).rolling("24h", min_periods=3).mean())
        .reset_index(level=0, drop=True)
    )
    rolling_std = (
        df.groupby("station_id", group_keys=False)
        .apply(lambda x: x.set_index("timestamp")[column]
               .shift(1).rolling("24h", min_periods=3).std())
        .reset_index(level=0, drop=True)
    )

    df[f"{column}_rolling_mean_24h"] = rolling_mean.values
    df[f"{column}_rolling_std_24h"] = rolling_std.values
    df[f"{column}_deviation_24h"] = df[column] - df[f"{column}_rolling_mean_24h"]
    std = df[f"{column}_rolling_std_24h"].replace(0, np.nan)
    df[f"{column}_zscore_24h"] = df[f"{column}_deviation_24h"] / std
    return df


def ensure_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds rolling features only for columns that don't already have them."""
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in CORE_FEATURES:
        if f"{col}_zscore_24h" not in df.columns:
            df = create_rolling_features(df, col)
    return df


# ---------------------------------------------------------------------
# 2. Statistical / threshold detector (baseline QC — not the final say)
# ---------------------------------------------------------------------
def statistical_detector(row) -> int:
    """Coarse physical-range check. NOTE: >40C is NOT flagged alone —
    Safdarjung legitimately hits ~44C in summer, so range is intentionally wide."""
    if row["temperature"] < -20 or row["temperature"] > 50:
        return 1
    if row["pressure"] < 950 or row["pressure"] > 1050:
        return 1
    if row["humidity"] < 0 or row["humidity"] > 100:
        return 1
    return 0


# ---------------------------------------------------------------------
# 3. Temporal detector (rolling z-score based)
# ---------------------------------------------------------------------
def temporal_detector(df: pd.DataFrame) -> pd.DataFrame:
    df["temporal_anomaly"] = (
        df[ZSCORE_COLS].abs().max(axis=1) > 3
    ).astype(int)
    df["temporal_score"] = (
        df[ZSCORE_COLS].abs().max(axis=1).clip(0, 6) / 6
    ).fillna(0)
    return df


# ---------------------------------------------------------------------
# 4. Isolation Forest (multivariate ML detector)
# ---------------------------------------------------------------------
def run_isolation_forest(df: pd.DataFrame, features=None,
                          contamination=0.05, n_estimators=200,
                          save_model_path=None, scaler_path=None):
    features = features or CORE_FEATURES
    X = df[features].copy()
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
    )
    iso_forest.fit(X_scaled)

    df["isolation_anomaly"] = (iso_forest.predict(X_scaled) == -1).astype(int)
    raw_score = -iso_forest.decision_function(X_scaled)

    min_s, max_s = raw_score.min(), raw_score.max()
    df["ml_anomaly_score"] = (raw_score - min_s) / (max_s - min_s + 1e-9)

    if save_model_path:
        joblib.dump(iso_forest, save_model_path)
    if scaler_path:
        joblib.dump(scaler, scaler_path)

    return df, iso_forest, scaler


# ---------------------------------------------------------------------
# 5. Ensemble score, severity, and human-readable reason
# ---------------------------------------------------------------------
def compute_ensemble_score(df: pd.DataFrame, w_temporal=0.4, w_ml=0.6) -> pd.DataFrame:
    df["satark_score"] = w_temporal * df["temporal_score"] + w_ml * df["ml_anomaly_score"]
    df["satark_prediction"] = (df["satark_score"] >= 0.5).astype(int)
    return df


def get_severity(score: float) -> str:
    if score < 0.30:
        return "Normal"
    elif score < 0.60:
        return "Low"
    elif score < 0.80:
        return "Medium"
    else:
        return "High"


def generate_reason(row) -> str:
    reasons = []
    if abs(row.get("temperature_zscore_24h", 0) or 0) > 3:
        reasons.append("Temperature deviates strongly from recent baseline")
    if abs(row.get("pressure_zscore_24h", 0) or 0) > 3:
        reasons.append("Pressure deviates strongly from recent baseline")
    if abs(row.get("humidity_zscore_24h", 0) or 0) > 3:
        reasons.append("Humidity deviates strongly from recent baseline")
    if row.get("ml_anomaly_score", 0) > 0.7:
        reasons.append("ML model identifies unusual multivariate pattern")
    if not reasons:
        return "No significant anomaly detected"
    return "; ".join(reasons)


# ---------------------------------------------------------------------
# 6. Full pipeline — this is the single function P4 / P3 should call
# ---------------------------------------------------------------------
def run_anomaly_pipeline(df: pd.DataFrame,
                          model_path=None,
                          scaler_path=None) -> pd.DataFrame:
    """
    Runs the complete P2 detection stack on a dataframe and returns it
    with all detection columns added. Safe to call on the synthetic
    fault dataset (leakage columns are ignored, never used as features).
    """
    df = df.copy()
    df = ensure_temporal_features(df)

    df["statistical_anomaly"] = df.apply(statistical_detector, axis=1)
    df = temporal_detector(df)
    df, _, _ = run_isolation_forest(
        df, save_model_path=model_path, scaler_path=scaler_path
    )
    df = compute_ensemble_score(df)

    df["severity"] = df["satark_score"].apply(get_severity)
    df["reason"] = df.apply(generate_reason, axis=1)

    return df


if __name__ == "__main__":
    # Quick standalone smoke test
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "processed_meteostat_data.csv"
    data = pd.read_csv(path)
    result = run_anomaly_pipeline(data)
    print(result[[
        "station_id", "timestamp", "temperature",
        "statistical_anomaly", "isolation_anomaly", "temporal_anomaly",
        "satark_score", "severity", "reason"
    ]].tail(10))
