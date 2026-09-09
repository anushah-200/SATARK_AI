
import pandas as pd


def classify_weather_vs_sensor(
    ml_anomaly,
    temporal_anomaly,
    station_value,
    neighbor_average,
    spatial_threshold=5.0
):
    """
    Classify an anomalous observation as:
    - Sensor Fault
    - Suspicious
    - Likely Weather Event

    Evidence is collected from:
    1. ML anomaly detection
    2. Temporal anomaly detection
    3. Spatial deviation from nearby stations
    """

    evidence = 0
    reasons = []

    # ML evidence
    if ml_anomaly:
        evidence += 1
        reasons.append("ML anomaly detected")

    # Temporal evidence
    if temporal_anomaly:
        evidence += 1
        reasons.append("Temporal deviation detected")

    # Spatial evidence
    if not pd.isna(neighbor_average):

        spatial_difference = abs(
            station_value - neighbor_average
        )

        if spatial_difference > spatial_threshold:
            evidence += 1
            reasons.append(
                "Value differs significantly from nearby stations"
            )

    # Final classification
    if evidence >= 3:
        classification = "Sensor Fault"

    elif evidence == 2:
        classification = "Suspicious"

    else:
        classification = "Likely Weather Event"

    return classification, evidence, reasons


def correct_value(
    original_value,
    neighbor_average,
    last_valid_value=None
):
    """
    Correct a faulty value.

    Priority:
    1. Neighbor average
    2. Last valid value
    3. Original value
    """

    if not pd.isna(neighbor_average):
        return neighbor_average

    if last_valid_value is not None:
        return last_valid_value

    return original_value
