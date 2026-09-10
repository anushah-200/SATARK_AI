
import pandas as pd


# ============================================================
# Meteorological Quality Control Thresholds
# ============================================================

TEMP_MIN = -50.0
TEMP_MAX = 60.0

HUMIDITY_MIN = 0.0
HUMIDITY_MAX = 100.0

PRESSURE_MIN = 870.0
PRESSURE_MAX = 1085.0


# ============================================================
# Individual QC Rules
# ============================================================

def check_temperature(value):
    """
    Flag temperature values outside the broad physical
    plausibility range.

    Returns:
        0 = normal
        1 = suspicious
    """
    if pd.isna(value):
        return 1

    return int(value < TEMP_MIN or value > TEMP_MAX)


def check_humidity(value):
    """
    Flag humidity values outside the physical range.

    Returns:
        0 = normal
        1 = suspicious
    """
    if pd.isna(value):
        return 1

    return int(value < HUMIDITY_MIN or value > HUMIDITY_MAX)


def check_pressure(value):
    """
    Flag pressure values outside the broad physical
    plausibility range.

    Returns:
        0 = normal
        1 = suspicious
    """
    if pd.isna(value):
        return 1

    return int(value < PRESSURE_MIN or value > PRESSURE_MAX)


# ============================================================
# Apply QC Rules to a DataFrame
# ============================================================

def apply_quality_control(df):
    """
    Apply basic meteorological QC rules to a weather DataFrame.

    Expected columns:
        temperature
        humidity
        pressure

    Adds:
        temperature_rule_flag
        humidity_rule_flag
        pressure_rule_flag
        qc_flag

    Returns:
        DataFrame with QC flags added.
    """

    df = df.copy()

    # Check required columns
    required_columns = [
        "temperature",
        "humidity",
        "pressure"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Apply individual rules
    df["temperature_rule_flag"] = (
        df["temperature"].apply(check_temperature)
    )

    df["humidity_rule_flag"] = (
        df["humidity"].apply(check_humidity)
    )

    df["pressure_rule_flag"] = (
        df["pressure"].apply(check_pressure)
    )

    # Overall QC flag
    df["qc_flag"] = (
        (df["temperature_rule_flag"] == 1)
        | (df["humidity_rule_flag"] == 1)
        | (df["pressure_rule_flag"] == 1)
    ).astype(int)

    return df
