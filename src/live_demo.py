"""
live_demo.py
------------
SATARK AI — historical / simulation inference layer for the Streamlit demo.

Uses saved models + in-memory station history. Does NOT:
  - refit Isolation Forest
  - retrain the ML corrector
  - modify batch P1/P2/P3 pipelines
  - write to any CSV outputs
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.anomaly_detector import get_severity
from src.person3_diagnosis_correction import (
    NORMAL,
    apply_ml_correction,
    correct_temperature,
    diagnose_faults,
)
from src.sensor_health import get_health_status, update_health_score


ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
P2_HISTORY_PATH = ROOT_DIR / "outputs" / "results" / "person2_day2_results.csv"

ISO_MODEL_PATH = MODELS_DIR / "isolation_forest.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ML_CORRECTOR_PATH = MODELS_DIR / "ml_corrector.pkl"

# Hourly history: enough for 24h baselines + robust MAD thresholds
MIN_HISTORY_ROWS = 48
DEFAULT_HISTORY_ROWS = 14 * 24  # ~14 days

IF_FEATURES = ["temperature", "pressure", "humidity"]

HEALTH_FAULT_MAP = {
    "SPIKE": "temperature_spike",
    "DRIFT": "temperature_drift",
    "FROZEN": "temperature_frozen",
    "MISSING": "temperature_missing",
    "NOISE": "multivariate_inconsistency",
    "SUSPICIOUS": "multivariate_inconsistency",
}


# ---------------------------------------------------------------------
# Model / history loaders (read-only)
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_models():
    import warnings

    missing = [
        p.name
        for p in (ISO_MODEL_PATH, SCALER_PATH, ML_CORRECTOR_PATH)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing model file(s): {', '.join(missing)} under {MODELS_DIR}"
        )

    # Saved artifacts may come from an older scikit-learn; load still works.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        isolation_forest = joblib.load(ISO_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        ml_corrector = joblib.load(ML_CORRECTOR_PATH)
    return isolation_forest, scaler, ml_corrector


@lru_cache(maxsize=1)
def _load_history_frame() -> pd.DataFrame:
    if not P2_HISTORY_PATH.exists():
        raise FileNotFoundError(f"History file not found: {P2_HISTORY_PATH}")

    cols = [
        "timestamp",
        "station_id",
        "station_name",
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
    ]
    df = pd.read_csv(P2_HISTORY_PATH, usecols=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce")
    for col in ("temperature", "humidity", "pressure", "wind_speed"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "station_id"]).sort_values(
        ["station_id", "timestamp"]
    )
    return df.reset_index(drop=True)


def list_demo_stations() -> pd.DataFrame:
    """Unique stations available for the simulation demo."""
    df = _load_history_frame()
    stations = (
        df.groupby("station_id", as_index=False)
        .agg(
            station_name=("station_name", "first"),
            last_timestamp=("timestamp", "max"),
            n_rows=("timestamp", "count"),
        )
        .sort_values("station_id")
    )
    return stations


def get_station_baseline(station_id: Union[int, str]) -> dict[str, Any]:
    """Latest historical reading for a station (for Normal preset defaults)."""
    df = _load_history_frame()
    sid = int(station_id)
    station_df = df[df["station_id"] == sid]
    if station_df.empty:
        raise ValueError(f"Unknown station_id: {station_id}")
    row = station_df.iloc[-1]
    return {
        "station_id": sid,
        "station_name": row.get("station_name"),
        "timestamp": row["timestamp"],
        "temperature": float(row["temperature"]) if pd.notna(row["temperature"]) else None,
        "pressure": float(row["pressure"]) if pd.notna(row["pressure"]) else None,
        "humidity": float(row["humidity"]) if pd.notna(row["humidity"]) else None,
        "wind_speed": float(row["wind_speed"]) if pd.notna(row["wind_speed"]) else 0.0,
        "history_rows": int(len(station_df)),
    }


# ---------------------------------------------------------------------
# Feature engineering (P2 notebook formulas, in-memory only)
# ---------------------------------------------------------------------

def _rolling_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any() or len(values) < 2:
        return np.nan
    return float(np.polyfit(np.arange(len(values)), values, 1)[0])


def _engineer_p2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the engineered columns expected by diagnose_faults / correction."""
    out = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    gtemp = out.groupby("station_id")["temperature"]

    # Past-only 24h baseline (hourly window ≈ 24 observations)
    shifted = gtemp.shift(1)
    roll_mean = (
        shifted.groupby(out["station_id"])
        .rolling(window=24, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    roll_std = (
        shifted.groupby(out["station_id"])
        .rolling(window=24, min_periods=3)
        .std()
        .reset_index(level=0, drop=True)
    )
    out["temperature_deviation_24h"] = out["temperature"] - roll_mean
    out["temperature_zscore_24h"] = out["temperature_deviation_24h"] / roll_std.replace(
        0, np.nan
    )

    out["temperature_change"] = gtemp.diff()
    out["absolute_temperature_change"] = out["temperature_change"].abs()

    out["temperature_std_6h"] = (
        gtemp.rolling(window=6, min_periods=2).std().reset_index(level=0, drop=True)
    )
    out["temperature_trend_6h"] = (
        gtemp.rolling(window=6, min_periods=6)
        .apply(_rolling_slope, raw=True)
        .reset_index(level=0, drop=True)
    )

    previous = gtemp.shift(1)
    out["previous_6h_median"] = (
        previous.groupby(out["station_id"])
        .rolling(window=6, min_periods=3)
        .median()
        .reset_index(level=0, drop=True)
    )
    out["local_temperature_residual"] = (
        out["temperature"] - out["previous_6h_median"]
    ).abs()

    out["previous_12h_median"] = (
        previous.groupby(out["station_id"])
        .rolling(window=12, min_periods=6)
        .median()
        .reset_index(level=0, drop=True)
    )
    out["local_residual_12h"] = (
        out["temperature"] - out["previous_12h_median"]
    ).abs()

    out["repeat_group"] = gtemp.transform(lambda x: (x != x.shift()).cumsum())
    out["repeat_length"] = (
        out.groupby(["station_id", "repeat_group"])["temperature"].transform("size")
    )
    out = out.drop(columns=["repeat_group"])
    return out


def _apply_detector_scores(
    df: pd.DataFrame,
    isolation_forest,
    scaler,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Attach P2-style detector columns using the SAVED Isolation Forest.
    Never calls run_isolation_forest() (which refits).
    """
    out = df.copy()

    temperature_rule = (
        (out["temperature"] < -50)
        | (out["temperature"] > 60)
        | out["temperature"].isna()
    )
    humidity_rule = (
        (out["humidity"] < 0) | (out["humidity"] > 100) | out["humidity"].isna()
    )
    pressure_rule = (
        (out["pressure"] < 870) | (out["pressure"] > 1085) | out["pressure"].isna()
    )
    out["rule_score"] = (
        temperature_rule | humidity_rule | pressure_rule
    ).astype(int)

    out["statistical_score"] = (
        (out["temperature_zscore_24h"].abs() > 3)
        | (out["temperature_deviation_24h"].abs() > 10)
    ).fillna(False).astype(int)

    feature_frame = out[IF_FEATURES].copy()
    feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))
    X_scaled = scaler.transform(feature_frame)
    iso_pred = isolation_forest.predict(X_scaled)
    raw_scores = -isolation_forest.decision_function(X_scaled)
    out["isolation_score"] = (iso_pred == -1).astype(int)

    sudden_change = out["absolute_temperature_change"] > 6
    frozen_pattern = (out["repeat_length"] >= 6) & (out["temperature_std_6h"] <= 1.0)
    extreme_trend = out["temperature_trend_6h"].abs() > 3
    out["temporal_score"] = (
        sudden_change.fillna(False)
        | frozen_pattern.fillna(False)
        | extreme_trend.fillna(False)
    ).astype(int)

    out["anomaly_score"] = (
        out["rule_score"]
        + out["statistical_score"]
        + out["isolation_score"]
        + out["temporal_score"]
    )
    return out, raw_scores


def _build_explanation(row: pd.Series, diagnosed_fault: str) -> str:
    parts = []
    if int(row.get("rule_score", 0) or 0) == 1:
        parts.append("Physical-range quality check flagged the reading")
    if int(row.get("statistical_score", 0) or 0) == 1:
        parts.append("Statistical deviation from the 24h station baseline")
    if int(row.get("isolation_score", 0) or 0) == 1:
        parts.append("Saved Isolation Forest flagged a multivariate anomaly")
    if int(row.get("temporal_score", 0) or 0) == 1:
        parts.append("Temporal pattern (sudden change / freeze / extreme trend)")

    if diagnosed_fault == "INSUFFICIENT CONTEXT":
        return "Insufficient station history or feature context for confident diagnosis"
    if diagnosed_fault == NORMAL:
        if not parts:
            return "No detector evidence of anomaly; reading consistent with recent station history"
        return (
            "Detectors raised weak/mixed signals but primary diagnosis remains NORMAL; "
            + "; ".join(parts)
        )
    if diagnosed_fault == "SUSPICIOUS":
        base = "Detector evidence without a confident specific fault type"
        return base + (": " + "; ".join(parts) if parts else "")
    label = f"Primary P3 diagnosis: {diagnosed_fault}"
    return label + (". " + "; ".join(parts) if parts else "")


def _insufficient_result(
    station_id,
    timestamp,
    temperature,
    reason: str,
) -> dict[str, Any]:
    return {
        "station_id": int(station_id) if station_id is not None else None,
        "timestamp": timestamp,
        "temperature_original": temperature,
        "anomaly_detected": False,
        "anomaly_score": None,
        "diagnosed_fault": "INSUFFICIENT CONTEXT",
        "diagnosis_confidence": None,
        "temperature_corrected": None,
        "correction_method": "none",
        "correction_applied": 0,
        "severity": "Unknown",
        "explanation": reason,
        "reason": reason,
        "ml_anomaly_score": None,
        "detector_scores": None,
        "sensor_health_score": None,
        "sensor_health_status": None,
        "pipeline_stages": {
            "quality_check": "skipped",
            "anomaly_detection": "skipped",
            "diagnosis": "INSUFFICIENT CONTEXT",
            "correction": "skipped",
            "sensor_health": "skipped",
        },
    }


# ---------------------------------------------------------------------
# Main demo API
# ---------------------------------------------------------------------

def run_live_demo(
    station_id: Union[int, str],
    timestamp: Optional[Union[str, pd.Timestamp]] = None,
    temperature: Optional[float] = None,
    pressure: Optional[float] = None,
    humidity: Optional[float] = None,
) -> dict[str, Any]:
    """
    Run a single simulated sensor reading through detection → diagnosis → correction
    using historical context and saved models only.
    """
    try:
        isolation_forest, scaler, ml_corrector = _load_models()
        history = _load_history_frame()
    except Exception as exc:
        return _insufficient_result(
            station_id, timestamp, temperature, f"Unable to load models/history: {exc}"
        )

    try:
        sid = int(station_id)
    except (TypeError, ValueError):
        return _insufficient_result(
            station_id, timestamp, temperature, f"Invalid station_id: {station_id}"
        )

    if temperature is None or pressure is None or humidity is None:
        return _insufficient_result(
            sid,
            timestamp,
            temperature,
            "temperature, pressure, and humidity are all required",
        )

    station_hist = history[history["station_id"] == sid].copy()
    if len(station_hist) < MIN_HISTORY_ROWS:
        return _insufficient_result(
            sid,
            timestamp,
            temperature,
            (
                f"Station {sid} has only {len(station_hist)} historical rows "
                f"(need ≥ {MIN_HISTORY_ROWS}) for temporal/diagnosis context"
            ),
        )

    # Keep a recent window only (in memory)
    window = station_hist.tail(DEFAULT_HISTORY_ROWS).copy()
    last_ts = window["timestamp"].iloc[-1]
    last_wind = window["wind_speed"].iloc[-1]
    station_name = window["station_name"].iloc[-1]

    if timestamp is None or (isinstance(timestamp, float) and np.isnan(timestamp)):
        demo_ts = last_ts + pd.Timedelta(hours=1)
    else:
        demo_ts = pd.to_datetime(timestamp, errors="coerce")
        if pd.isna(demo_ts):
            return _insufficient_result(
                sid, timestamp, temperature, f"Invalid timestamp: {timestamp}"
            )
        # Avoid colliding with an existing row: place after last history point
        if demo_ts <= last_ts:
            demo_ts = last_ts + pd.Timedelta(hours=1)

    demo_row = {
        "timestamp": demo_ts,
        "station_id": sid,
        "station_name": station_name,
        "temperature": float(temperature),
        "humidity": float(humidity),
        "pressure": float(pressure),
        "wind_speed": float(last_wind) if pd.notna(last_wind) else 0.0,
    }
    working = pd.concat([window, pd.DataFrame([demo_row])], ignore_index=True)

    working = _engineer_p2_features(working)
    working, raw_if_scores = _apply_detector_scores(working, isolation_forest, scaler)

    demo_idx = working.index[-1]
    demo_features_ok = pd.notna(working.loc[demo_idx, "previous_6h_median"]) and pd.notna(
        working.loc[demo_idx, "temperature_zscore_24h"]
    )
    if not demo_features_ok:
        return _insufficient_result(
            sid,
            demo_ts,
            temperature,
            "Could not compute required temporal baselines for the simulated reading",
        )

    # Primary P3 diagnosis + rule correction on the in-memory window only
    diagnosed = diagnose_faults(working)
    corrected = correct_temperature(diagnosed)
    # Ensure correction_method exists before ML pass
    if "correction_method" not in corrected.columns:
        corrected["correction_method"] = "none"
    final_df = apply_ml_correction(corrected, ml_corrector)

    row = final_df.iloc[-1]
    diagnosed_fault = str(row.get("diagnosed_fault", NORMAL))
    confidence = row.get("diagnosis_confidence", None)
    if pd.notna(confidence):
        confidence = float(confidence)
    else:
        confidence = None

    anomaly_score = int(row["anomaly_score"]) if pd.notna(row.get("anomaly_score")) else 0
    ml_score_raw = float(raw_if_scores[-1])
    # Map IF raw score to ~[0,1] for severity helper without inventing detector votes
    ml_anomaly_score = float(1.0 / (1.0 + np.exp(-ml_score_raw)))
    severity_input = max(ml_anomaly_score, anomaly_score / 4.0)
    severity = get_severity(severity_input) if diagnosed_fault != NORMAL else "Normal"

    correction_applied = int(row.get("correction_applied", 0) or 0)
    correction_method = str(row.get("correction_method", "none") or "none")
    temperature_corrected = row.get("temperature_corrected", None)
    if pd.notna(temperature_corrected):
        temperature_corrected = float(temperature_corrected)
    else:
        temperature_corrected = None

    explanation = _build_explanation(row, diagnosed_fault)
    anomaly_detected = diagnosed_fault not in {NORMAL, "INSUFFICIENT CONTEXT"}

    health_key = HEALTH_FAULT_MAP.get(diagnosed_fault)
    if anomaly_detected and health_key:
        health_score = update_health_score(100, health_key)
    else:
        health_score = 100
    health_status = get_health_status(health_score)

    qc_flag = "flagged" if int(row.get("rule_score", 0) or 0) == 1 else "passed"
    detection_flag = "anomaly" if anomaly_score >= 1 else "normal"

    return {
        "station_id": sid,
        "station_name": station_name,
        "timestamp": demo_ts,
        "temperature_original": float(temperature),
        "pressure": float(pressure),
        "humidity": float(humidity),
        "anomaly_detected": bool(anomaly_detected),
        "anomaly_score": anomaly_score,
        "diagnosed_fault": diagnosed_fault,
        "diagnosis_confidence": confidence,
        "temperature_corrected": temperature_corrected,
        "correction_method": correction_method,
        "correction_applied": correction_applied,
        "severity": severity,
        "explanation": explanation,
        "reason": explanation,
        "ml_anomaly_score": round(ml_anomaly_score, 4),
        "detector_scores": {
            "rule_score": int(row.get("rule_score", 0) or 0),
            "statistical_score": int(row.get("statistical_score", 0) or 0),
            "isolation_score": int(row.get("isolation_score", 0) or 0),
            "temporal_score": int(row.get("temporal_score", 0) or 0),
        },
        "sensor_health_score": health_score,
        "sensor_health_status": health_status,
        "pipeline_stages": {
            "quality_check": qc_flag,
            "anomaly_detection": detection_flag,
            "diagnosis": diagnosed_fault,
            "correction": correction_method if correction_applied else "none",
            "sensor_health": health_status,
        },
    }


if __name__ == "__main__":
    stations = list_demo_stations()
    print("Stations:", stations.to_string(index=False))
    sid = int(stations.iloc[0]["station_id"])
    base = get_station_baseline(sid)
    print("\nBaseline:", base)

    print("\n--- Normal ---")
    print(
        run_live_demo(
            sid,
            None,
            base["temperature"],
            base["pressure"],
            base["humidity"],
        )
    )

    print("\n--- Spike 55C ---")
    print(
        run_live_demo(
            sid,
            None,
            55.0,
            base["pressure"],
            base["humidity"],
        )
    )
