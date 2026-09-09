import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.person3_pipeline import process_observations
from src.spatial_analysis import create_distance_matrix


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM PROFESSIONAL STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main {
        padding-top: 1.5rem;
    }

    h1 {
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    h2 {
        font-weight: 600;
    }

    h3 {
        font-weight: 500;
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        padding: 15px;
    }

    .status-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        padding: 25px;
        margin-top: 30px;
        border-top: 1px solid rgba(128,128,128,0.25);
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.title("SkyGuard AI")
st.subheader("Predictive Anomaly Detection and Sensor Monitoring")

st.caption(
    "Historical anomaly detection dashboard with replay and simulation-based monitoring."
)


# ============================================================
# LOAD DATA
# ============================================================

RESULT_FILE = ROOT_DIR / "outputs" / "results" / "anomaly_detection_results.csv"

if not RESULT_FILE.exists():
    st.error(
        f"Result file not found: {RESULT_FILE}"
    )
    st.stop()


@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    data["is_anomaly"] = pd.to_numeric(
        data["is_anomaly"],
        errors="coerce"
    ).fillna(0)

    return data


try:
    df = load_data(str(RESULT_FILE))
except Exception as e:
    st.error(f"Unable to load result file: {e}")
    st.stop()


dashboard_df = df.copy()


# ============================================================
# PERSON 3 INTEGRATION
# ============================================================

@st.cache_data
def run_person3_pipeline(data):

    p3_df = data.copy()

    p3_df["timestamp"] = pd.to_datetime(
        p3_df["timestamp"],
        errors="coerce"
    )

    p3_df["ml_anomaly"] = (
        pd.to_numeric(
            p3_df["isolation_anomaly"],
            errors="coerce"
        )
        .fillna(0)
        .astype(bool)
    )

    p3_df["temporal_anomaly"] = (
        pd.to_numeric(
            p3_df["temporal_anomaly"],
            errors="coerce"
        )
        .fillna(0)
        .astype(bool)
    )

    station_info = (
        p3_df[
            [
                "station_id",
                "latitude",
                "longitude"
            ]
        ]
        .drop_duplicates("station_id")
        .reset_index(drop=True)
    )

    distance_matrix = create_distance_matrix(
        station_info
    )

    observations = p3_df[
        p3_df["is_anomaly"] == 1
    ][
        [
            "station_id",
            "timestamp",
            "ml_anomaly",
            "temporal_anomaly"
        ]
    ].copy()

    if len(observations) == 0:
        return pd.DataFrame()

    results = process_observations(
        df=p3_df,
        observations=observations,
        distance_matrix=distance_matrix,
        parameter="temperature"
    )

    return results


with st.spinner("Integrating spatial diagnosis and sensor health..."):

    try:
        person3_results = run_person3_pipeline(
            dashboard_df
        )
    except Exception as e:
        person3_results = pd.DataFrame()
        st.warning(
            f"Person 3 integration unavailable: {e}"
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Controls")

st.sidebar.markdown("### Station Selection")

station_options = ["All Stations"]

if "station_name" in dashboard_df.columns:

    station_names = sorted(
        dashboard_df["station_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    station_options.extend(station_names)


selected_station = st.sidebar.selectbox(
    "Select Station",
    station_options
)


# ============================================================
# STATION FILTER
# ============================================================

if selected_station == "All Stations":

    filtered_df = dashboard_df.copy()

else:

    filtered_df = dashboard_df[
        dashboard_df["station_name"].astype(str)
        == selected_station
    ].copy()


if len(filtered_df) == 0:

    st.warning(
        "No data available for the selected station."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown("---")

st.markdown(
    f"### Monitoring: {selected_station}"
)


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_records = len(filtered_df)

anomaly_count = int(
    filtered_df["is_anomaly"].sum()
)

normal_count = (
    total_records - anomaly_count
)

if total_records > 0:

    anomaly_rate = (
        anomaly_count /
        total_records
    ) * 100

else:

    anomaly_rate = 0


# ============================================================
# SYSTEM OVERVIEW
# ============================================================

st.markdown("## System Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Records",
        f"{total_records:,}"
    )

with col2:

    st.metric(
        "Anomalies Detected",
        f"{anomaly_count:,}"
    )

with col3:

    st.metric(
        "Normal Records",
        f"{normal_count:,}"
    )

with col4:

    st.metric(
        "Anomaly Rate",
        f"{anomaly_rate:.2f}%"
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

if anomaly_count == 0:

    st.success(
        "System Status: Normal"
    )

elif anomaly_rate < 5:

    st.warning(
        f"System Status: Monitoring Required — "
        f"{anomaly_count:,} anomalies detected"
    )

else:

    st.error(
        f"System Status: Elevated Anomaly Activity — "
        f"{anomaly_count:,} anomalies detected"
    )


# ============================================================
# LATEST ANOMALY
# ============================================================

st.markdown("## Latest Anomaly Alert")

anomaly_df = filtered_df[
    filtered_df["is_anomaly"] == 1
].copy()


if len(anomaly_df) > 0:

    anomaly_df = anomaly_df.sort_values(
        "timestamp",
        ascending=False
    )

    latest = anomaly_df.iloc[0]

    alert_col1, alert_col2 = st.columns(2)

    with alert_col1:

        st.error(
            "Anomaly Detected"
        )

        st.write(
            f"**Time:** {latest['timestamp']}"
        )

        if "station_name" in latest:

            st.write(
                f"**Station:** "
                f"{latest['station_name']}"
            )

        if "severity" in latest:

            st.write(
                f"**Severity:** "
                f"{latest['severity']}"
            )

        if "fault_type" in latest:

            st.write(
                f"**Fault Type:** "
                f"{latest['fault_type']}"
            )

    with alert_col2:

        if "satark_score" in latest:

            score = pd.to_numeric(
                latest["satark_score"],
                errors="coerce"
            )

            st.metric(
                "Satark Score",
                f"{score:.3f}"
                if pd.notna(score)
                else "N/A"
            )

        if "satark_prediction" in latest:

            st.write(
                f"**Prediction:** "
                f"{latest['satark_prediction']}"
            )

        if "reason" in latest:

            reason = str(
                latest["reason"]
            )

            if reason.lower() != "nan":

                st.info(
                    f"**Reason:** {reason}"
                )

else:

    st.success(
        "No anomalies found for the selected station."
    )


# ============================================================
# PERSON 3 DIAGNOSIS
# ============================================================

st.markdown("## Spatial Diagnosis and Sensor Health")

if len(person3_results) > 0:

    p3_display = person3_results.copy()

    if selected_station != "All Stations":

        if "station_id" in filtered_df.columns:

            selected_ids = (
                filtered_df["station_id"]
                .dropna()
                .unique()
                .tolist()
            )

            p3_display = p3_display[
                p3_display["station_id"]
                .isin(selected_ids)
            ]

    if len(p3_display) > 0:

        latest_p3 = p3_display.sort_values(
            "timestamp",
            ascending=False
        ).iloc[0]

        p3_col1, p3_col2, p3_col3, p3_col4 = st.columns(4)

        with p3_col1:

            if "classification" in latest_p3:

                st.metric(
                    "Diagnosis",
                    str(
                        latest_p3["classification"]
                    )
                )

        with p3_col2:

            if "health_score" in latest_p3:

                health_score = pd.to_numeric(
                    latest_p3["health_score"],
                    errors="coerce"
                )

                st.metric(
                    "Sensor Health",
                    f"{health_score:.0f}/100"
                    if pd.notna(health_score)
                    else "N/A"
                )

        with p3_col3:

            if "neighbor_average" in latest_p3:

                neighbor_average = pd.to_numeric(
                    latest_p3["neighbor_average"],
                    errors="coerce"
                )

                st.metric(
                    "Neighbor Average",
                    f"{neighbor_average:.2f}"
                    if pd.notna(neighbor_average)
                    else "N/A"
                )

        with p3_col4:

            if "spatial_difference" in latest_p3:

                spatial_difference = pd.to_numeric(
                    latest_p3["spatial_difference"],
                    errors="coerce"
                )

                st.metric(
                    "Spatial Difference",
                    f"{spatial_difference:.2f}"
                    if pd.notna(spatial_difference)
                    else "N/A"
                )

        diagnosis_columns = [
            "timestamp",
            "station_id",
            "parameter",
            "classification",
            "evidence_score",
            "fault_type",
            "corrected_value",
            "health_score",
            "health_status",
            "reasons"
        ]

        available_diagnosis_columns = [
            col
            for col in diagnosis_columns
            if col in p3_display.columns
        ]

        st.dataframe(
            p3_display[
                available_diagnosis_columns
            ].sort_values(
                "timestamp",
                ascending=False
            ).head(10),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No Person 3 diagnosis is available for this station."
        )

else:

    st.info(
        "Person 3 diagnosis results are not available."
    )


# ============================================================
# RECENT DATA
# ============================================================

st.markdown("## Recent Sensor Data")

recent_data = filtered_df.sort_values(
    "timestamp",
    ascending=False
).head(10)


display_columns = [
    "timestamp",
    "station_name",
    "temperature",
    "humidity",
    "pressure",
    "satark_score",
    "satark_prediction",
    "severity"
]


available_columns = [
    col
    for col in display_columns
    if col in recent_data.columns
]


st.dataframe(
    recent_data[
        available_columns
    ],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SENSOR MONITORING
# ============================================================

st.markdown("## Sensor Monitoring")

plot_df = filtered_df.sort_values(
    "timestamp"
).copy()

plot_anomalies = plot_df[
    plot_df["is_anomaly"] == 1
]


# ============================================================
# TEMPERATURE
# ============================================================

if "temperature" in plot_df.columns:

    st.markdown("### Temperature")

    temperature_fig = px.line(
        plot_df,
        x="timestamp",
        y="temperature",
        title="Temperature Over Time",
        labels={
            "timestamp": "Time",
            "temperature": "Temperature"
        }
    )

    if len(plot_anomalies) > 0:

        temperature_anomalies = (
            plot_anomalies[
                [
                    "timestamp",
                    "temperature"
                ]
            ]
            .dropna()
        )

        temperature_fig.add_scatter(
            x=temperature_anomalies["timestamp"],
            y=temperature_anomalies["temperature"],
            mode="markers",
            name="Detected Anomaly",
            marker=dict(
                size=8,
                symbol="x"
            )
        )

    st.plotly_chart(
        temperature_fig,
        use_container_width=True
    )


# ============================================================
# HUMIDITY
# ============================================================

if "humidity" in plot_df.columns:

    st.markdown("### Humidity")

    humidity_fig = px.line(
        plot_df,
        x="timestamp",
        y="humidity",
        title="Humidity Over Time",
        labels={
            "timestamp": "Time",
            "humidity": "Humidity"
        }
    )

    if len(plot_anomalies) > 0:

        humidity_anomalies = (
            plot_anomalies[
                [
                    "timestamp",
                    "humidity"
                ]
            ]
            .dropna()
        )

        humidity_fig.add_scatter(
            x=humidity_anomalies["timestamp"],
            y=humidity_anomalies["humidity"],
            mode="markers",
            name="Detected Anomaly",
            marker=dict(
                size=8,
                symbol="x"
            )
        )

    st.plotly_chart(
        humidity_fig,
        use_container_width=True
    )


# ============================================================
# PRESSURE
# ============================================================

if "pressure" in plot_df.columns:

    st.markdown("### Pressure")

    pressure_fig = px.line(
        plot_df,
        x="timestamp",
        y="pressure",
        title="Pressure Over Time",
        labels={
            "timestamp": "Time",
            "pressure": "Pressure"
        }
    )

    if len(plot_anomalies) > 0:

        pressure_anomalies = (
            plot_anomalies[
                [
                    "timestamp",
                    "pressure"
                ]
            ]
            .dropna()
        )

        pressure_fig.add_scatter(
            x=pressure_anomalies["timestamp"],
            y=pressure_anomalies["pressure"],
            mode="markers",
            name="Detected Anomaly",
            marker=dict(
                size=8,
                symbol="x"
            )
        )

    st.plotly_chart(
        pressure_fig,
        use_container_width=True
    )


# ============================================================
# FAULT TYPE DISTRIBUTION
# ============================================================

if "fault_type" in filtered_df.columns:

    st.markdown("## Fault Type Distribution")

    fault_df = filtered_df[
        filtered_df["is_anomaly"] == 1
    ].copy()

    if len(fault_df) > 0:

        fault_counts = (
            fault_df["fault_type"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .reset_index()
        )

        fault_counts.columns = [
            "fault_type",
            "count"
        ]

        fault_fig = px.bar(
            fault_counts,
            x="fault_type",
            y="count",
            title="Detected Anomalies by Fault Type",
            labels={
                "fault_type": "Fault Type",
                "count": "Number of Anomalies"
            }
        )

        st.plotly_chart(
            fault_fig,
            use_container_width=True
        )

    else:

        st.info(
            "No fault-type data available for anomalies."
        )


# ============================================================
# SEVERITY DISTRIBUTION
# ============================================================

severity_column = None

if "severity" in filtered_df.columns:

    severity_column = "severity"

elif "fault_severity" in filtered_df.columns:

    severity_column = "fault_severity"


if severity_column is not None:

    st.markdown("## Severity Distribution")

    severity_df = filtered_df[
        filtered_df["is_anomaly"] == 1
    ].copy()

    if len(severity_df) > 0:

        severity_counts = (
            severity_df[
                severity_column
            ]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .reset_index()
        )

        severity_counts.columns = [
            "severity",
            "count"
        ]

        severity_fig = px.pie(
            severity_counts,
            names="severity",
            values="count",
            title="Anomaly Severity Distribution"
        )

        st.plotly_chart(
            severity_fig,
            use_container_width=True
        )

    else:

        st.info(
            "No severity data available."
        )


# ============================================================
# SENSOR HEALTH SUMMARY
# ============================================================

st.markdown("## Sensor Health Summary")

if "station_name" in filtered_df.columns:

    health_data = []

    for station in (
        filtered_df["station_name"]
        .dropna()
        .unique()
    ):

        station_data = filtered_df[
            filtered_df["station_name"]
            == station
        ]

        records = len(station_data)

        anomalies = int(
            station_data[
                "is_anomaly"
            ].sum()
        )

        if records > 0:

            anomaly_percentage = (
                anomalies /
                records
            ) * 100

        else:

            anomaly_percentage = 0

        if anomaly_percentage < 2:

            health_status = "Healthy"

        elif anomaly_percentage < 5:

            health_status = "Watch"

        else:

            health_status = "At Risk"

        health_data.append(
            {
                "Station": station,
                "Records": records,
                "Anomalies": anomalies,
                "Anomaly Rate (%)":
                    round(
                        anomaly_percentage,
                        2
                    ),
                "Health Status":
                    health_status
            }
        )

    health_df = pd.DataFrame(
        health_data
    )

    st.dataframe(
        health_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Sensor health is represented as a risk score based on "
        "observed anomaly activity and does not claim confirmed "
        "physical hardware failure."
    )


# ============================================================
# STATION COMPARISON
# ============================================================

if "station_name" in filtered_df.columns:

    st.markdown("## Station Comparison")

    station_comparison = (
        filtered_df
        .groupby("station_name")
        .agg(
            Total_Records=(
                "is_anomaly",
                "size"
            ),
            Anomalies=(
                "is_anomaly",
                "sum"
            )
        )
        .reset_index()
    )

    station_comparison[
        "Anomaly_Rate"
    ] = (
        station_comparison["Anomalies"]
        /
        station_comparison["Total_Records"]
    ) * 100

    station_comparison[
        "Anomaly_Rate"
    ] = station_comparison[
        "Anomaly_Rate"
    ].round(2)

    station_fig = px.bar(
        station_comparison,
        x="station_name",
        y="Anomaly_Rate",
        title="Anomaly Rate by Station",
        labels={
            "station_name": "Station",
            "Anomaly_Rate": "Anomaly Rate (%)"
        }
    )

    st.plotly_chart(
        station_fig,
        use_container_width=True
    )


# ============================================================
# SATARK SCORE
# ============================================================

if "satark_score" in filtered_df.columns:

    st.markdown("## Satark Anomaly Score")

    score_df = filtered_df.sort_values(
        "timestamp"
    ).copy()

    score_fig = px.line(
        score_df,
        x="timestamp",
        y="satark_score",
        title="Satark Score Over Time",
        labels={
            "timestamp": "Time",
            "satark_score": "Satark Score"
        }
    )

    st.plotly_chart(
        score_fig,
        use_container_width=True
    )


# ============================================================
# MODEL SIGNALS
# ============================================================

st.markdown("## Detection Signals")

signal_columns = [
    "statistical_anomaly",
    "temporal_anomaly",
    "isolation_anomaly"
]

available_signal_columns = [
    col
    for col in signal_columns
    if col in filtered_df.columns
]


if available_signal_columns:

    signal_counts = {}

    for column in available_signal_columns:

        signal_counts[column] = int(
            pd.to_numeric(
                filtered_df[column],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

    signal_df = pd.DataFrame(
        {
            "Detection Signal":
                list(
                    signal_counts.keys()
                ),
            "Detected Records":
                list(
                    signal_counts.values()
                )
        }
    )

    signal_fig = px.bar(
        signal_df,
        x="Detection Signal",
        y="Detected Records",
        title="Model Detection Signals"
    )

    st.plotly_chart(
        signal_fig,
        use_container_width=True
    )


# ============================================================
# HISTORICAL REPLAY NOTICE
# ============================================================

st.markdown("---")

st.info(
    "Demo Mode: The current dashboard uses historical data "
    "replay and simulation to demonstrate anomaly monitoring. "
    "It should not be presented as a live AWS data stream."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <b>SkyGuard AI</b><br>
        Predictive Anomaly Detection and Sensor Monitoring<br>
        P4 — Streamlit and Plotly System Integration
    </div>
    """,
    unsafe_allow_html=True
)