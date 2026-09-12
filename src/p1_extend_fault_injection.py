"""
p1_extend_fault_injection.py
------------------------------
SATARK AI — P1 fix: extend fault injection beyond temperature-only.

Uses the SAME 5 fault types P1 already established for temperature
(spike, frozen, drift, noise, missing) and the SAME severity magnitudes
already defined in the original project spec, applied to humidity and
pressure. Existing 20 temperature faults are left completely untouched —
new faults are injected only into time windows that don't overlap them.

Input : processed_meteostat_data.csv (clean, all 5 stations)
        p2_day2_handoff.csv (to read existing temperature fault windows)
Output: p1_day3_extended_handoff.csv (same schema + humidity/pressure faults)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# same severity scale already defined in the original project spec
SEVERITY_MAGNITUDE = {
    "humidity": {"mild": 10, "moderate": 25, "severe": 40},
    "pressure": {"mild": 5, "moderate": 10, "severe": 20},
}
DRIFT_STEP = {"mild": 0.2, "moderate": 0.5, "severe": 1.0}
DURATIONS = [1, 2, 3, 4, 6]  # hours, same range P1 used for temperature

injected_regions = []  # (station_id, start_time, end_time) — global, shared across variables


def region_overlaps(station_id, start_time, end_time):
    for r in injected_regions:
        if r["station_id"] != station_id:
            continue
        if start_time < r["end_time"] and end_time > r["start_time"]:
            return True
    return False


def register_region(station_id, start_time, end_time):
    injected_regions.append({"station_id": station_id, "start_time": start_time, "end_time": end_time})


def seed_existing_temperature_faults(handoff_path):
    """Load the 20 existing temperature fault windows so new faults never overlap them."""
    df = pd.read_csv(handoff_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    faults = df[df["is_anomaly"] == 1]
    for station_id, g in faults.groupby("station_id"):
        # treat each contiguous block as one region, with 1h padding on each side
        g = g.sort_values("timestamp")
        start = g["timestamp"].iloc[0] - pd.Timedelta(hours=1)
        end = g["timestamp"].iloc[0]
        for ts in g["timestamp"].iloc[1:]:
            if ts - end <= pd.Timedelta(hours=1):
                end = ts
            else:
                register_region(station_id, start, end + pd.Timedelta(hours=1))
                start = ts - pd.Timedelta(hours=1)
                end = ts
        register_region(station_id, start, end + pd.Timedelta(hours=1))


def inject_spike(df, station_id, start_time, duration, variable, severity):
    end_time = start_time + pd.Timedelta(hours=duration)
    mask = (df["station_id"] == station_id) & (df["timestamp"] >= start_time) & (df["timestamp"] < end_time)
    magnitude = SEVERITY_MAGNITUDE[variable][severity]
    df.loc[mask, variable] += magnitude
    df.loc[mask, "is_anomaly"] = 1
    df.loc[mask, "fault_type"] = f"{variable}_spike"
    df.loc[mask, "fault_severity"] = severity
    return df


def inject_frozen(df, station_id, start_time, duration, variable, severity):
    end_time = start_time + pd.Timedelta(hours=duration)
    mask = (df["station_id"] == station_id) & (df["timestamp"] >= start_time) & (df["timestamp"] < end_time)
    indices = df.index[mask]
    if len(indices) == 0:
        return df
    frozen_value = df.loc[indices[0], variable]
    df.loc[indices, variable] = frozen_value
    df.loc[indices, "is_anomaly"] = 1
    df.loc[indices, "fault_type"] = f"{variable}_frozen"
    df.loc[indices, "fault_severity"] = severity
    return df


def inject_drift(df, station_id, start_time, duration, variable, severity):
    end_time = start_time + pd.Timedelta(hours=duration)
    mask = (df["station_id"] == station_id) & (df["timestamp"] >= start_time) & (df["timestamp"] < end_time)
    indices = df.index[mask]
    if len(indices) == 0:
        return df
    step = DRIFT_STEP[severity]
    for i, idx in enumerate(indices):
        df.loc[idx, variable] += (i + 1) * step
    df.loc[indices, "is_anomaly"] = 1
    df.loc[indices, "fault_type"] = f"{variable}_drift"
    df.loc[indices, "fault_severity"] = severity
    return df


def inject_noise(df, station_id, start_time, duration, variable, severity):
    """Alternating irregular perturbations — matches the pattern P1 already used for temperature_noise."""
    end_time = start_time + pd.Timedelta(hours=duration)
    mask = (df["station_id"] == station_id) & (df["timestamp"] >= start_time) & (df["timestamp"] < end_time)
    indices = df.index[mask]
    if len(indices) == 0:
        return df
    magnitude = SEVERITY_MAGNITUDE[variable][severity]
    signs = np.array([1, -1, 1, 1, -1, 1])[:len(indices)]
    perturbation = signs * magnitude * np.random.uniform(0.4, 1.0, size=len(indices))
    df.loc[indices, variable] += perturbation
    df.loc[indices, "is_anomaly"] = 1
    df.loc[indices, "fault_type"] = f"{variable}_noise"
    df.loc[indices, "fault_severity"] = severity
    return df


def inject_missing(df, station_id, start_time, duration, variable, severity):
    end_time = start_time + pd.Timedelta(hours=duration)
    mask = (df["station_id"] == station_id) & (df["timestamp"] >= start_time) & (df["timestamp"] < end_time)
    df.loc[mask, variable] = np.nan
    df.loc[mask, "is_anomaly"] = 1
    df.loc[mask, "fault_type"] = f"{variable}_missing"
    df.loc[mask, "fault_severity"] = severity
    return df


INJECTORS = {
    "spike": inject_spike, "frozen": inject_frozen,
    "drift": inject_drift, "noise": inject_noise, "missing": inject_missing,
}


def generate_extended_faults(df, n_per_variable=5):
    """Injects n_per_variable faults each for humidity and pressure, one of each of the
    5 fault types (spike/frozen/drift/noise/missing), spread across stations/times,
    never overlapping existing temperature faults or each other."""
    stations = df["station_id"].unique()
    fault_types = list(INJECTORS.keys())
    severities = ["mild", "moderate", "severe"]
    log = []

    for variable in ["humidity", "pressure"]:
        for fault_type in fault_types:
            placed = False
            attempts = 0
            while not placed and attempts < 200:
                attempts += 1
                station_id = np.random.choice(stations)
                duration = np.random.choice(DURATIONS)
                severity = np.random.choice(severities)
                station_rows = df[df["station_id"] == station_id]
                # keep well inside the series so duration doesn't run off the edge
                start_idx = np.random.randint(200, len(station_rows) - 200)
                start_time = station_rows["timestamp"].iloc[start_idx]
                end_time = start_time + pd.Timedelta(hours=int(duration))

                if region_overlaps(station_id, start_time, end_time):
                    continue

                df = INJECTORS[fault_type](df, station_id, start_time, int(duration), variable, severity)
                register_region(station_id, start_time, end_time)
                log.append({
                    "variable": variable, "fault_type": fault_type, "station_id": station_id,
                    "start_time": start_time, "duration": duration, "severity": severity,
                })
                placed = True
            if not placed:
                print(f"WARNING: could not place {variable}_{fault_type} after 200 attempts")

    return df, pd.DataFrame(log)


if __name__ == "__main__":
    clean = pd.read_csv("processed_meteostat_data.csv")
    clean["timestamp"] = pd.to_datetime(clean["timestamp"])

    # start from the existing extended dataset so temperature faults are preserved as-is
    existing = pd.read_csv("p2_day2_handoff.csv")
    existing["timestamp"] = pd.to_datetime(existing["timestamp"])

    for col in ["original_humidity", "original_pressure"]:
        if col not in existing.columns:
            existing[col] = existing[col.replace("original_", "")]

    seed_existing_temperature_faults("p2_day2_handoff.csv")

    extended, log = generate_extended_faults(existing)

    print("New faults injected:")
    print(log)
    print()
    print("fault_type distribution (temperature + humidity + pressure):")
    print(extended["fault_type"].value_counts())
    print()
    print("total is_anomaly:", extended["is_anomaly"].sum())

    extended.to_csv("p1_day3_extended_handoff.csv", index=False)
    log.to_csv("p1_day3_injection_log.csv", index=False)
    print("saved p1_day3_extended_handoff.csv")
