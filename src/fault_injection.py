import pandas as pd
import numpy as np



def inject_temperature_spike(
    df,
    station_id,
    row_index,
    spike_value=55.0,
    severity="severe"
):
    """
    Inject a temperature spike into one observation.

    Parameters
    ----------
    df : pandas.DataFrame
        Input weather dataset.

    station_id : int
        Station where the fault should be injected.

    row_index : int
        Index of the row within the selected station.

    spike_value : float
        Temperature value to inject.

    severity : str
        Fault severity label.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the injected temperature spike.
    """

    df = df.copy()

    # Check required columns
    required_columns = [
        "station_id",
        "temperature"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Select rows belonging to the requested station
    station_rows = df.index[
        df["station_id"] == station_id
    ].tolist()

    # Check station exists
    if not station_rows:
        raise ValueError(
            f"Station {station_id} was not found in the dataset."
        )

    # Check row_index is valid
    if row_index < 0 or row_index >= len(station_rows):
        raise IndexError(
            f"row_index must be between 0 and "
            f"{len(station_rows) - 1}."
        )

    # Get actual DataFrame index
    target_index = station_rows[row_index]

    # Preserve original temperature
    if "original_temperature" not in df.columns:
        df["original_temperature"] = df["temperature"]

    # Inject spike
    df.loc[target_index, "temperature"] = spike_value

    # Add/overwrite fault labels
    if "fault_type" not in df.columns:
        df["fault_type"] = "normal"

    if "fault_severity" not in df.columns:
        df["fault_severity"] = "none"

    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = 0

    df.loc[target_index, "fault_type"] = "temperature_spike"
    df.loc[target_index, "fault_severity"] = severity
    df.loc[target_index, "is_anomaly"] = 1

    return df


def inject_missing(
    df,
    station_id,
    row_index,
    parameter="temperature",
    severity="moderate"
):
    """
    Inject a missing-value fault into one observation.

    The original value is preserved in the corresponding
    original_<parameter> column, while the observed value
    is replaced with NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        Input weather dataset.

    station_id : int
        Station where the fault should be injected.

    row_index : int
        Row position within the selected station.

    parameter : str
        Weather parameter to make missing.
        Examples: temperature, humidity, pressure.

    severity : str
        Fault severity label.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the injected missing value.
    """

    df = df.copy()

    # Check required columns
    if "station_id" not in df.columns:
        raise ValueError("Column 'station_id' is missing.")

    if parameter not in df.columns:
        raise ValueError(
            f"Parameter '{parameter}' is not present in the dataset."
        )

    # Find rows belonging to the selected station
    station_rows = df.index[
        df["station_id"] == station_id
    ].tolist()

    if not station_rows:
        raise ValueError(
            f"Station {station_id} was not found in the dataset."
        )

    # Check row position
    if row_index < 0 or row_index >= len(station_rows):
        raise IndexError(
            f"row_index must be between 0 and "
            f"{len(station_rows) - 1}."
        )

    target_index = station_rows[row_index]

    # Preserve original value
    original_column = f"original_{parameter}"

    if original_column not in df.columns:
        df[original_column] = df[parameter]

    # Inject missing value
    df.loc[target_index, parameter] = np.nan

    # Create fault columns if needed
    if "fault_type" not in df.columns:
        df["fault_type"] = "normal"

    if "fault_severity" not in df.columns:
        df["fault_severity"] = "none"

    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = 0

    # Label fault
    df.loc[target_index, "fault_type"] = f"{parameter}_missing"
    df.loc[target_index, "fault_severity"] = severity
    df.loc[target_index, "is_anomaly"] = 1

    return df


def inject_frozen(
    df,
    station_id,
    start_row,
    duration=6,
    parameter="temperature",
    severity="moderate"
):
    """
    Inject a frozen/stuck sensor fault over consecutive observations.

    The first value in the selected range is repeated for the
    remaining observations.

    Parameters
    ----------
    df : pandas.DataFrame
        Input weather dataset.

    station_id : int
        Station where the fault should be injected.

    start_row : int
        Starting row position within the selected station.

    duration : int
        Number of consecutive observations affected.

    parameter : str
        Weather parameter to freeze.

    severity : str
        Fault severity label.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the frozen sensor fault.
    """

    df = df.copy()

    # Validate columns
    if "station_id" not in df.columns:
        raise ValueError("Column 'station_id' is missing.")

    if parameter not in df.columns:
        raise ValueError(
            f"Parameter '{parameter}' is not present in the dataset."
        )

    # Get station rows
    station_rows = df.index[
        df["station_id"] == station_id
    ].tolist()

    if not station_rows:
        raise ValueError(
            f"Station {station_id} was not found in the dataset."
        )

    # Validate duration
    if duration <= 0:
        raise ValueError("duration must be greater than 0.")

    # Validate start position
    if start_row < 0 or start_row >= len(station_rows):
        raise IndexError(
            f"start_row must be between 0 and "
            f"{len(station_rows) - 1}."
        )

    # Make sure the sequence fits
    end_row = start_row + duration

    if end_row > len(station_rows):
        raise IndexError(
            f"Not enough observations. "
            f"Only {len(station_rows) - start_row} observations "
            f"are available from start_row={start_row}."
        )

    # Preserve original values
    original_column = f"original_{parameter}"

    if original_column not in df.columns:
        df[original_column] = df[parameter]

    # Get the actual DataFrame indices
    target_indices = station_rows[start_row:end_row]

    # Use the first value as the frozen sensor value
    frozen_value = df.loc[target_indices[0], parameter]

    # Freeze the sensor
    df.loc[target_indices, parameter] = frozen_value

    # Create fault columns if needed
    if "fault_type" not in df.columns:
        df["fault_type"] = "normal"

    if "fault_severity" not in df.columns:
        df["fault_severity"] = "none"

    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = 0

    # Label affected observations
    df.loc[target_indices, "fault_type"] = f"{parameter}_frozen"
    df.loc[target_indices, "fault_severity"] = severity
    df.loc[target_indices, "is_anomaly"] = 1

    return df


def inject_drift(
    df,
    station_id,
    start_row,
    duration=6,
    parameter="temperature",
    drift_per_step=1.0,
    severity="moderate"
):
    """
    Inject a gradual sensor drift over consecutive observations.

    Each subsequent reading is shifted further from the
    original value by drift_per_step.

    Parameters
    ----------
    df : pandas.DataFrame
        Input weather dataset.

    station_id : int
        Station where the drift should be injected.

    start_row : int
        Starting row position within the selected station.

    duration : int
        Number of consecutive observations affected.

    parameter : str
        Weather parameter to modify.

    drift_per_step : float
        Amount of additional drift applied at each step.

    severity : str
        Fault severity label.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the injected drift.
    """

    df = df.copy()

    # Validate columns
    if "station_id" not in df.columns:
        raise ValueError("Column 'station_id' is missing.")

    if parameter not in df.columns:
        raise ValueError(
            f"Parameter '{parameter}' is not present in the dataset."
        )

    # Get station rows
    station_rows = df.index[
        df["station_id"] == station_id
    ].tolist()

    if not station_rows:
        raise ValueError(
            f"Station {station_id} was not found in the dataset."
        )

    # Validate duration
    if duration <= 0:
        raise ValueError("duration must be greater than 0.")

    if start_row < 0 or start_row >= len(station_rows):
        raise IndexError(
            f"start_row must be between 0 and "
            f"{len(station_rows) - 1}."
        )

    end_row = start_row + duration

    if end_row > len(station_rows):
        raise IndexError(
            f"Not enough observations for the requested duration."
        )

    # Preserve original values
    original_column = f"original_{parameter}"

    if original_column not in df.columns:
        df[original_column] = df[parameter]

    # Select target rows
    target_indices = station_rows[start_row:end_row]

    # Apply gradually increasing drift
    for step, idx in enumerate(target_indices):
        original_value = df.loc[idx, original_column]

        if pd.isna(original_value):
            continue

        df.loc[idx, parameter] = (
            original_value + (drift_per_step * step)
        )

    # Create fault columns if needed
    if "fault_type" not in df.columns:
        df["fault_type"] = "normal"

    if "fault_severity" not in df.columns:
        df["fault_severity"] = "none"

    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = 0

    # Label affected rows
    df.loc[target_indices, "fault_type"] = f"{parameter}_drift"
    df.loc[target_indices, "fault_severity"] = severity
    df.loc[target_indices, "is_anomaly"] = 1

    return df


def inject_noise(
    df,
    station_id,
    start_row,
    duration=6,
    parameter="temperature",
    noise_std=2.0,
    random_state=42,
    severity="mild"
):
    """
    Inject random sensor noise over consecutive observations.

    Parameters
    ----------
    df : pandas.DataFrame
        Input weather dataset.

    station_id : int
        Station where the fault should be injected.

    start_row : int
        Starting row position within the selected station.

    duration : int
        Number of consecutive observations affected.

    parameter : str
        Weather parameter to modify.

    noise_std : float
        Standard deviation of the random noise.

    random_state : int
        Seed for reproducible noise.

    severity : str
        Fault severity label.

    Returns
    -------
    pandas.DataFrame
        DataFrame with injected sensor noise.
    """

    df = df.copy()

    # Validate columns
    if "station_id" not in df.columns:
        raise ValueError("Column 'station_id' is missing.")

    if parameter not in df.columns:
        raise ValueError(
            f"Parameter '{parameter}' is not present in the dataset."
        )

    # Get station rows
    station_rows = df.index[
        df["station_id"] == station_id
    ].tolist()

    if not station_rows:
        raise ValueError(
            f"Station {station_id} was not found in the dataset."
        )

    # Validate duration
    if duration <= 0:
        raise ValueError("duration must be greater than 0.")

    if start_row < 0 or start_row >= len(station_rows):
        raise IndexError(
            f"start_row must be between 0 and "
            f"{len(station_rows) - 1}."
        )

    end_row = start_row + duration

    if end_row > len(station_rows):
        raise IndexError(
            "Not enough observations for the requested duration."
        )

    # Preserve original values
    original_column = f"original_{parameter}"

    if original_column not in df.columns:
        df[original_column] = df[parameter]

    # Select target rows
    target_indices = station_rows[start_row:end_row]

    # Reproducible random generator
    rng = np.random.default_rng(random_state)

    # Apply random noise
    for idx in target_indices:

        original_value = df.loc[idx, original_column]

        if pd.isna(original_value):
            continue

        noise = rng.normal(
            loc=0.0,
            scale=noise_std
        )

        df.loc[idx, parameter] = original_value + noise

    # Create fault columns if needed
    if "fault_type" not in df.columns:
        df["fault_type"] = "normal"

    if "fault_severity" not in df.columns:
        df["fault_severity"] = "none"

    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = 0

    # Label affected observations
    df.loc[target_indices, "fault_type"] = f"{parameter}_noise"
    df.loc[target_indices, "fault_severity"] = severity
    df.loc[target_indices, "is_anomaly"] = 1

    return df
