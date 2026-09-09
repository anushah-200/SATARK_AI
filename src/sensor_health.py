
# Penalty applied to sensor health score for different fault types

FAULT_PENALTIES = {
    "temperature_spike": 15,
    "temperature_drift": 10,
    "temperature_frozen": 12,
    "temperature_drop": 10,

    "humidity_spike": 8,
    "humidity_drift": 8,
    "humidity_frozen": 10,
    "humidity_drop": 8,

    "pressure_spike": 8,
    "pressure_drift": 8,
    "pressure_frozen": 10,
    "pressure_drop": 8,

    "temperature_missing": 5,
    "humidity_missing": 5,
    "pressure_missing": 5,

    "multivariate_inconsistency": 10
}


def update_health_score(
    current_score,
    fault_type
):
    """
    Reduce sensor health score according to fault type.

    Score is kept between 0 and 100.
    """

    penalty = FAULT_PENALTIES.get(
        fault_type,
        5
    )

    new_score = max(
        0,
        current_score - penalty
    )

    return new_score


def get_health_status(score):
    """
    Convert numerical health score into a status.

    80-100  -> GREEN
    50-79   -> YELLOW
    0-49    -> RED
    """

    if score >= 80:
        return "GREEN"

    elif score >= 50:
        return "YELLOW"

    return "RED"
