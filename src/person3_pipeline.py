
import pandas as pd

from src.spatial_analysis import get_neighbor_consensus
from src.diagnosis import classify_weather_vs_sensor, correct_value
from src import sensor_health


def diagnose_sensor_observation(
    df,
    station_id,
    timestamp,
    parameter,
    distance_matrix,
    ml_anomaly=False,
    temporal_anomaly=False,
    current_health=100,
    last_valid_value=None,
    max_distance_km=100,
    spatial_threshold=5.0
):
    """
    Complete Person-3 pipeline for one observation.

    Spatial analysis
        ↓
    Diagnosis
        ↓
    Correction
        ↓
    Sensor health
    """

    # --------------------------------------------------
    # 1. Get target observation
    # --------------------------------------------------

    target = df[
        (df["station_id"] == station_id) &
        (df["timestamp"] == timestamp)
    ]

    if target.empty:
        raise ValueError(
            "No observation found for the given station and timestamp."
        )

    station_value = target.iloc[0][parameter]

    # --------------------------------------------------
    # 2. Spatial analysis
    # --------------------------------------------------

    consensus = get_neighbor_consensus(
        df=df,
        station_id=station_id,
        timestamp=timestamp,
        parameter=parameter,
        distance_matrix=distance_matrix,
        max_distance_km=max_distance_km
    )

    neighbor_average = consensus["neighbor_average"]

    neighbor_count = consensus["neighbor_count"]

    normal_neighbors = consensus["normal_neighbors"]

    anomalous_neighbors = consensus["anomalous_neighbors"]

    # --------------------------------------------------
    # 3. Diagnosis
    # --------------------------------------------------

    classification, evidence_score, reasons = (
        classify_weather_vs_sensor(
            ml_anomaly=ml_anomaly,
            temporal_anomaly=temporal_anomaly,
            station_value=station_value,
            neighbor_average=neighbor_average,
            spatial_threshold=spatial_threshold
        )
    )

    # --------------------------------------------------
    # 4. Correction
    # --------------------------------------------------

    if classification == "Sensor Fault":

        corrected_value = correct_value(
            original_value=station_value,
            neighbor_average=neighbor_average,
            last_valid_value=last_valid_value
        )

    else:

        corrected_value = station_value

    # --------------------------------------------------
    # 5. Determine fault type
    # --------------------------------------------------

    fault_type = None

    if classification == "Sensor Fault":

        if parameter == "temperature":

            if station_value > 40:
                fault_type = "temperature_spike"

            elif station_value < -20:
                fault_type = "temperature_drop"

            else:
                fault_type = "temperature_spike"

        elif parameter == "humidity":

            fault_type = "humidity_spike"

        elif parameter == "pressure":

            fault_type = "pressure_spike"

    # --------------------------------------------------
    # 6. Update sensor health
    # --------------------------------------------------

    new_health = current_health

    if fault_type is not None:

        new_health = sensor_health.update_health_score(
            current_score=current_health,
            fault_type=fault_type
        )

    health_status = sensor_health.get_health_status(
        new_health
    )

    # --------------------------------------------------
    # 7. Spatial difference
    # --------------------------------------------------

    if pd.isna(neighbor_average):

        spatial_difference = None

    else:

        spatial_difference = abs(
            station_value - neighbor_average
        )

    # --------------------------------------------------
    # 8. Final result
    # --------------------------------------------------

    result = {
        "station_id": station_id,
        "timestamp": timestamp,
        "parameter": parameter,

        "original_value": station_value,

        "neighbor_count": neighbor_count,
        "normal_neighbors": normal_neighbors,
        "anomalous_neighbors": anomalous_neighbors,

        "neighbor_average": neighbor_average,
        "spatial_difference": spatial_difference,

        "ml_anomaly": ml_anomaly,
        "temporal_anomaly": temporal_anomaly,

        "classification": classification,
        "evidence_score": evidence_score,
        "reasons": reasons,

        "fault_type": fault_type,

        "corrected_value": corrected_value,

        "health_score": new_health,
        "health_status": health_status
    }

    return result


def process_observations(
    df,
    observations,
    distance_matrix,
    ml_anomaly_col="ml_anomaly",
    temporal_anomaly_col="temporal_anomaly",
    parameter="temperature",
    current_health=100,
    max_distance_km=100,
    spatial_threshold=5.0
):
    """
    Process multiple observations through the Person-3 pipeline.

    Returns a DataFrame containing:
    spatial analysis,
    diagnosis,
    correction,
    and sensor health.
    """

    results = []

    health_scores = {}

    for _, row in observations.iterrows():

        station_id = row["station_id"]
        timestamp = row["timestamp"]

        station_health = health_scores.get(
            station_id,
            current_health
        )

        ml_anomaly = bool(
            row.get(ml_anomaly_col, False)
        )

        temporal_anomaly = bool(
            row.get(temporal_anomaly_col, False)
        )

        result = diagnose_sensor_observation(
            df=df,
            station_id=station_id,
            timestamp=timestamp,
            parameter=parameter,
            distance_matrix=distance_matrix,
            ml_anomaly=ml_anomaly,
            temporal_anomaly=temporal_anomaly,
            current_health=station_health,
            max_distance_km=max_distance_km,
            spatial_threshold=spatial_threshold
        )

        health_scores[station_id] = result["health_score"]

        results.append(result)

    return pd.DataFrame(results)
