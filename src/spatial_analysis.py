
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two latitude/longitude
    coordinates in kilometers.
    """

    R = 6371.0

    lat1, lon1, lat2, lon2 = map(
        radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return R * c


def create_distance_matrix(stations):
    """
    Create a distance matrix containing the distance
    in kilometers between every pair of stations.

    stations must contain:
        station_id
        latitude
        longitude
    """

    station_ids = stations["station_id"].tolist()

    distance_matrix = pd.DataFrame(
        index=station_ids,
        columns=station_ids,
        dtype=float
    )

    for _, station_a in stations.iterrows():

        for _, station_b in stations.iterrows():

            distance = haversine_distance(
                station_a["latitude"],
                station_a["longitude"],
                station_b["latitude"],
                station_b["longitude"]
            )

            distance_matrix.loc[
                station_a["station_id"],
                station_b["station_id"]
            ] = distance

    return distance_matrix


def get_neighbors(
    station_id,
    distance_matrix,
    max_distance_km=100
):
    """
    Return stations within max_distance_km.

    The target station itself is excluded.
    """

    distances = distance_matrix.loc[station_id]

    neighbors = (
        distances[
            (distances > 0) &
            (distances <= max_distance_km)
        ]
        .sort_values()
    )

    return neighbors


def get_neighbor_average(
    df,
    station_id,
    timestamp,
    parameter,
    distance_matrix,
    max_distance_km=100
):
    """
    Calculate the average value of a parameter
    from nearby stations at the same timestamp.
    """

    neighbors = get_neighbors(
        station_id,
        distance_matrix,
        max_distance_km
    )

    if len(neighbors) == 0:
        return np.nan

    neighbor_ids = neighbors.index.tolist()

    subset = df[
        (df["timestamp"] == timestamp) &
        (df["station_id"].isin(neighbor_ids))
    ]

    values = subset[parameter].dropna()

    if len(values) == 0:
        return np.nan

    return values.mean()


def calculate_spatial_deviation(
    station_value,
    neighbor_average
):
    """
    Calculate the absolute difference between
    the station value and neighboring average.
    """

    if pd.isna(neighbor_average):
        return np.nan

    return abs(
        station_value - neighbor_average
    )


def get_neighbor_consensus(
    df,
    station_id,
    timestamp,
    parameter,
    distance_matrix,
    max_distance_km=100
):
    """
    Check whether nearby stations agree with each other
    and whether they are flagged as anomalous.
    """

    neighbors = get_neighbors(
        station_id,
        distance_matrix,
        max_distance_km
    )

    if len(neighbors) == 0:
        return {
            "neighbor_count": 0,
            "normal_neighbors": 0,
            "anomalous_neighbors": 0,
            "neighbor_average": np.nan,
            "neighbor_values": []
        }

    neighbor_ids = neighbors.index.tolist()

    neighbor_data = df[
        (df["timestamp"] == timestamp) &
        (df["station_id"].isin(neighbor_ids))
    ].copy()

    neighbor_data = neighbor_data.dropna(
        subset=[parameter]
    )

    if len(neighbor_data) == 0:
        return {
            "neighbor_count": 0,
            "normal_neighbors": 0,
            "anomalous_neighbors": 0,
            "neighbor_average": np.nan,
            "neighbor_values": []
        }

    neighbor_values = neighbor_data[parameter].tolist()

    if "is_anomaly" in neighbor_data.columns:

        anomalous_neighbors = int(
            (neighbor_data["is_anomaly"] == 1).sum()
        )

        normal_neighbors = int(
            (neighbor_data["is_anomaly"] == 0).sum()
        )

    else:

        anomalous_neighbors = 0
        normal_neighbors = len(neighbor_data)

    return {
        "neighbor_count": len(neighbor_data),
        "normal_neighbors": normal_neighbors,
        "anomalous_neighbors": anomalous_neighbors,
        "neighbor_average": np.mean(neighbor_values),
        "neighbor_values": neighbor_values
    }
