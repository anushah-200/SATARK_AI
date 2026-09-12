"""
spatial_consistency.py
------------------------
SATARK AI — Spatial consistency check (weather event vs sensor fault).

This is the project's original differentiator from a fixed-threshold system,
from the very first proposal:

    Station A = 55C, neighbors B/C/D = 28-30C  -> SENSOR FAULT
    Station A = 45C, neighbors B/C/D = 44-46C  -> GENUINE WEATHER EVENT

For each flagged row, compares the station's reading against the other
stations' readings at the SAME timestamp. If the reading is close to what
neighbors report, it's a genuine (if unusual) weather event -- not just a
random guess, all 5 stations independently seeing similar values is itself
strong evidence. If it's wildly different from every neighbor, it's much
more likely a local sensor problem.

Input : any P3 output (person3_diagnosis_correction_eval.csv) for the
        flagged station, PLUS the clean multi-station data
        (processed_meteostat_data.csv) to look up neighbor readings.
"""

import numpy as np
import pandas as pd


def build_station_pivot(all_stations_path="processed_meteostat_data.csv", variable="temperature"):
    """Wide table: one row per timestamp, one column per station_id."""
    df = pd.read_csv(all_stations_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    pivot = df.pivot_table(index="timestamp", columns="station_id", values=variable)
    return pivot


def classify_spatial_consistency(flagged_df, pivot, station_col="station_id",
                                  variable_col="temperature", threshold=5.0):
    """
    For each row in flagged_df, look up all OTHER stations' readings at the
    same timestamp and classify:
      - WEATHER_EVENT   : reading is close to the neighbor average (within
                           threshold) -- consistent with a real regional
                           phenomenon, not a local fault
      - SENSOR_FAULT     : reading diverges sharply from every neighbor --
                           no regional support for a reading this extreme
      - INSUFFICIENT_DATA: fewer than 2 neighbors have data at this timestamp

    threshold (degC for temperature) is the max allowed deviation from the
    neighbor average before a reading is treated as spatially inconsistent.
    Default 5C matches the kind of spatial spread already visible between
    these 5 stations on ordinary days (they're 30-150km apart).
    """
    df = flagged_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    neighbor_avg = []
    neighbor_count = []
    for _, row in df.iterrows():
        ts, station = row["timestamp"], row[station_col]
        if ts not in pivot.index:
            neighbor_avg.append(np.nan)
            neighbor_count.append(0)
            continue
        readings = pivot.loc[ts].drop(labels=[station], errors="ignore").dropna()
        neighbor_avg.append(readings.mean() if len(readings) > 0 else np.nan)
        neighbor_count.append(len(readings))

    df["neighbor_avg_temperature"] = neighbor_avg
    df["neighbor_count"] = neighbor_count
    df["spatial_deviation"] = (df[variable_col] - df["neighbor_avg_temperature"]).abs()

    def classify(row):
        if row["neighbor_count"] < 2 or pd.isna(row["spatial_deviation"]):
            return "INSUFFICIENT_DATA"
        return "WEATHER_EVENT" if row["spatial_deviation"] <= threshold else "SENSOR_FAULT"

    df["spatial_classification"] = df.apply(classify, axis=1)
    return df


def apply_spatial_confidence_boost(df, boost=0.2):
    """
    Use spatial corroboration to INCREASE diagnosis_confidence when the two
    signals agree (SENSOR_FAULT + a non-NORMAL diagnosis), rather than using
    spatial_classification as a pass/fail gate. This preserves recall on
    subtle faults (frozen/drift/noise) that spatial checks can't see, while
    giving a real confidence lift to cases like the 55C spike where both
    signals agree independently -- which is itself meaningful evidence.
    """
    df = df.copy()
    if "spatial_classification" not in df.columns:
        return df
    corroborated = (df["diagnosed_fault"] != "NORMAL") & (df["spatial_classification"] == "SENSOR_FAULT")
    df.loc[corroborated, "diagnosis_confidence"] = (
        df.loc[corroborated, "diagnosis_confidence"] + boost
    ).clip(upper=1.0)
    return df


if __name__ == "__main__":
    import sys

    p3_path = sys.argv[1] if len(sys.argv) > 1 else "person3_diagnosis_correction_eval.csv"
    p3 = pd.read_csv(p3_path)
    p3["timestamp"] = pd.to_datetime(p3["timestamp"])

    pivot = build_station_pivot("processed_meteostat_data.csv", variable="temperature")

    flagged = p3[p3["diagnosed_fault"] != "NORMAL"].copy()
    result = classify_spatial_consistency(flagged, pivot, variable_col="temperature", threshold=5.0)

    print(f"Classified {len(result)} flagged rows:")
    print(result["spatial_classification"].value_counts())
    print()

    # validate against the real ground truth, if present
    if "is_anomaly" in result.columns:
        truth = result[result["is_anomaly"] == 1]
        print("On the 20 TRUE injected faults specifically:")
        print(truth[["timestamp", "fault_type", "temperature", "neighbor_avg_temperature",
                      "spatial_deviation", "spatial_classification"]].to_string(index=False))

        # IMPORTANT: do not use spatial_classification as a strict AND-gate over
        # diagnosed_fault. It correctly rules the 55C spike a SENSOR_FAULT (large
        # magnitude, spatially implausible) but FROZEN/DRIFT/NOISE faults don't
        # look spatially unusual by nature -- gating on this would drop recall
        # from 55% to 5%, discarding real anomalies just because they're subtle.
        # Use it as a confidence/explanation signal ALONGSIDE diagnosis, not a
        # replacement filter -- see the module docstring and handover doc.
        print()
        print("NOTE: spatial_classification is a corroborating signal, not a filter.")
        print("Do not AND-gate it against diagnosed_fault -- see docstring.")

    out_cols = list(p3.columns) + ["neighbor_avg_temperature", "neighbor_count",
                                    "spatial_deviation", "spatial_classification"]
    full = p3.merge(
        result[["timestamp", "station_id", "neighbor_avg_temperature", "neighbor_count",
                "spatial_deviation", "spatial_classification"]],
        on=["timestamp", "station_id"], how="left"
    )
    full.to_csv("person3_with_spatial_check.csv", index=False)
    print("\nSaved person3_with_spatial_check.csv")
