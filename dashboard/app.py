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
# CUSTOM STYLING
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
    "Historical anomaly detection dashboard with diagnosis, correction "
    "and sensor monitoring."
)


# ============================================================
# FILE PATHS
# ============================================================

P3_FILE = ROOT_DIR / "results" / "person3_diagnosis_correction.csv"

METADATA_FILE = (
    ROOT_DIR
    / "outputs"
    / "results"
    / "anomaly_detection_results.csv"
)


# ============================================================
# LOAD P3 DATA
# ============================================================

if not P3_FILE.exists():
    st.error(
        f"P3 result file not found: {P3_FILE}"
    )
    st.info(
        "Expected file: results/person3_diagnosis_correction.csv"
    )
    st.stop()


@st.cache_data
def load_p3_data(file_path):
    data = pd.read_csv(file_path)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    numeric_columns = [
        "temperature",
        "temperature_corrected",
        "temperature_corrected_rule_only",
        "temperature_corrected_ml",
        "rule_score",
        "statistical_score",
        "isolation_score",
        "temporal_score",
        "anomaly_score",
        "diagnosis_confidence",
        "correction_applied"
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    return data


try:
    df = load_p3_data(str(P3_FILE))
except Exception as e:
    st.error(f"Unable to load P3 result file: {e}")
    st.stop()


# ============================================================
# OPTIONAL STATION METADATA
# ============================================================

@st.cache_data
def load_metadata(file_path):
    if not file_path.exists():
        return pd.DataFrame()

    try:
        metadata = pd.read_csv(file_path)

        useful_columns = [
            "timestamp",
            "station_id",
            "station_name",
            "latitude",
            "longitude",
            "humidity",
            "pressure",
            "wind_speed",
            "wind_direction"
        ]

        available = [
            col for col in useful_columns
            if col in metadata.columns
        ]

        return metadata[available].copy()

    except Exception:
        return pd.DataFrame()


metadata_df = load_metadata(METADATA_FILE)


# ============================================================
# MERGE STATION / WEATHER METADATA
# ============================================================

dashboard_df = df.copy()

if not metadata_df.empty:

    metadata_df["timestamp"] = pd.to_datetime(
        metadata_df["timestamp"],
        errors="coerce"
    )

    metadata_columns = [
        "timestamp",
        "station_id"
    ]

    extra_columns = [
        "station_name",
        "latitude",
        "longitude",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_direction"
    ]

    available_extra = [
        col for col in extra_columns
        if col in metadata_df.columns
    ]

    metadata_small = (
        metadata_df[
            metadata_columns + available_extra
        ]
        .drop_duplicates(
            ["timestamp", "station_id"]
        )
    )

    dashboard_df = dashboard_df.merge(
        metadata_small,
        on=["timestamp", "station_id"],
        how="left"
    )


# ============================================================
# CREATE DASHBOARD FIELDS
# ============================================================

dashboard_df["diagnosed_fault"] = (
    dashboard_df["diagnosed_fault"]
    .fillna("NORMAL")
    .astype(str)
)

dashboard_df["is_anomaly"] = (
    dashboard_df["diagnosed_fault"]
    .str.upper()
    .ne("NORMAL")
).astype(int)

dashboard_df["correction_applied"] = (
    pd.to_numeric(
        dashboard_df["correction_applied"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)

dashboard_df["diagnosis_confidence"] = (
    pd.to_numeric(
        dashboard_df["diagnosis_confidence"],
        errors="coerce"
    )
)

dashboard_df["anomaly_score"] = (
    pd.to_numeric(
        dashboard_df["anomaly_score"],
        errors="coerce"
    )
)


# ============================================================
# STATION LABEL
# ============================================================

if "station_name" not in dashboard_df.columns:
    dashboard_df["station_name"] = (
        "Station "
        + dashboard_df["station_id"].astype(str)
    )

dashboard_df["station_name"] = (
    dashboard_df["station_name"]
    .fillna(
        "Station "
        + dashboard_df["station_id"].astype(str)
    )
    .astype(str)
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Controls")

st.sidebar.markdown("### Station Selection")

station_options = ["All Stations"]

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
        dashboard_df["station_name"]
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

correction_count = int(
    filtered_df["correction_applied"].sum()
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
        "Diagnosed Anomalies",
        f"{anomaly_count:,}"
    )

with col3:

    st.metric(
        "Normal Records",
        f"{normal_count:,}"
    )

with col4:

    st.metric(
        "Corrections Applied",
        f"{correction_count:,}"
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
        f"{anomaly_count:,} diagnosed anomalies"
    )

else:

    st.error(
        f"System Status: Elevated Anomaly Activity — "
        f"{anomaly_count:,} diagnosed anomalies"
    )


# ============================================================
# LATEST DIAGNOSIS / ALERT
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

        st.write(
            f"**Station:** {latest['station_name']}"
        )

        st.write(
            f"**Fault Type:** "
            f"{latest['diagnosed_fault']}"
        )

        confidence = pd.to_numeric(
            latest["diagnosis_confidence"],
            errors="coerce"
        )

        st.write(
            f"**Diagnosis Confidence:** "
            f"{confidence:.2f}"
            if pd.notna(confidence)
            else "**Diagnosis Confidence:** N/A"
        )

    with alert_col2:

        original_temp = pd.to_numeric(
            latest["temperature"],
            errors="coerce"
        )

        corrected_temp = pd.to_numeric(
            latest["temperature_corrected"],
            errors="coerce"
        )

        st.metric(
            "Original Temperature",
            f"{original_temp:.2f} °C"
            if pd.notna(original_temp)
            else "N/A"
        )

        st.metric(
            "Corrected Temperature",
            f"{corrected_temp:.2f} °C"
            if pd.notna(corrected_temp)
            else "N/A"
        )

        correction_method = str(
            latest.get(
                "correction_method",
                "N/A"
            )
        )

        st.write(
            f"**Correction Method:** "
            f"{correction_method}"
        )

        if int(
            latest.get(
                "correction_applied",
                0
            )
        ) == 1:

            st.success(
                "Correction Applied"
            )

        else:

            st.info(
                "No correction applied"
            )

else:

    st.success(
        "No diagnosed anomalies found for the selected station."
    )


# ============================================================
# DIAGNOSIS SUMMARY
# ============================================================

st.markdown("## Diagnosis Summary")

diagnosis_counts = (
    filtered_df["diagnosed_fault"]
    .value_counts()
    .reset_index()
)

diagnosis_counts.columns = [
    "diagnosed_fault",
    "count"
]

diagnosis_fig = px.bar(
    diagnosis_counts,
    x="diagnosed_fault",
    y="count",
    title="Diagnosis Distribution",
    labels={
        "diagnosed_fault": "Diagnosis",
        "count": "Number of Records"
    }
)

st.plotly_chart(
    diagnosis_fig,
    use_container_width=True
)


# ============================================================
# RECENT DATA
# ============================================================

st.markdown("## Recent Sensor Data")

recent_data = (
    filtered_df
    .sort_values(
        "timestamp",
        ascending=False
    )
    .head(10)
)

display_columns = [
    "timestamp",
    "station_name",
    "temperature",
    "temperature_corrected",
    "diagnosed_fault",
    "diagnosis_confidence",
    "correction_method",
    "correction_applied"
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
# TEMPERATURE MONITORING
# ============================================================

st.markdown("## Sensor Monitoring")

plot_df = (
    filtered_df
    .sort_values("timestamp")
    .copy()
)


if "temperature" in plot_df.columns:

    st.markdown("### Temperature Before and After Correction")

    temperature_fig = px.line(
        plot_df,
        x="timestamp",
        y=[
            "temperature",
            "temperature_corrected"
        ],
        title="Temperature Correction",
        labels={
            "timestamp": "Time",
            "value": "Temperature (°C)",
            "variable": "Series"
        }
    )

    anomaly_points = plot_df[
        plot_df["is_anomaly"] == 1
    ].copy()

    if len(anomaly_points) > 0:

        temperature_fig.add_scatter(
            x=anomaly_points["timestamp"],
            y=anomaly_points["temperature"],
            mode="markers",
            name="Diagnosed Anomaly",
            marker=dict(
                size=9,
                symbol="x"
            )
        )

    st.plotly_chart(
        temperature_fig,
        use_container_width=True
    )


# ============================================================
# CORRECTION METHODS
# ============================================================

st.markdown("## Correction Methods")

if "correction_method" in filtered_df.columns:

    method_df = (
        filtered_df[
            filtered_df["correction_applied"] == 1
        ]
        .copy()
    )

    if len(method_df) > 0:

        method_counts = (
            method_df["correction_method"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .reset_index()
        )

        method_counts.columns = [
            "correction_method",
            "count"
        ]

        method_fig = px.bar(
            method_counts,
            x="correction_method",
            y="count",
            title="Applied Correction Methods",
            labels={
                "correction_method": "Correction Method",
                "count": "Corrections Applied"
            }
        )

        st.plotly_chart(
            method_fig,
            use_container_width=True
        )

    else:

        st.info(
            "No corrections were applied in the selected data."
        )


# ============================================================
# FAULT TYPE DISTRIBUTION
# ============================================================

st.markdown("## Fault Type Distribution")

fault_df = filtered_df[
    filtered_df["diagnosed_fault"].str.upper() != "NORMAL"
].copy()

if len(fault_df) > 0:

    fault_counts = (
        fault_df["diagnosed_fault"]
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
        title="Diagnosed Fault Types",
        labels={
            "fault_type": "Fault Type",
            "count": "Number of Records"
        }
    )

    st.plotly_chart(
        fault_fig,
        use_container_width=True
    )

else:

    st.info(
        "No diagnosed fault types available."
    )


# ============================================================
# DIAGNOSIS CONFIDENCE
# ============================================================

st.markdown("## Diagnosis Confidence")

confidence_df = filtered_df[
    filtered_df["diagnosis_confidence"].notna()
].copy()

if len(confidence_df) > 0:

    confidence_fig = px.histogram(
        confidence_df,
        x="diagnosis_confidence",
        nbins=20,
        title="Diagnosis Confidence Distribution",
        labels={
            "diagnosis_confidence": "Confidence"
        }
    )

    st.plotly_chart(
        confidence_fig,
        use_container_width=True
    )

else:

    st.info(
        "Diagnosis confidence data is not available."
    )


# ============================================================
# ANOMALY SCORE
# ============================================================

st.markdown("## Anomaly Score")

score_df = plot_df[
    plot_df["anomaly_score"].notna()
].copy()

if len(score_df) > 0:

    score_fig = px.line(
        score_df,
        x="timestamp",
        y="anomaly_score",
        title="Anomaly Score Over Time",
        labels={
            "timestamp": "Time",
            "anomaly_score": "Anomaly Score"
        }
    )

    st.plotly_chart(
        score_fig,
        use_container_width=True
    )

else:

    st.info(
        "Anomaly score data is not available."
    )


# ============================================================
# MODEL SIGNALS
# ============================================================

st.markdown("## Detection Signals")

signal_columns = [
    "rule_score",
    "statistical_score",
    "isolation_score",
    "temporal_score"
]

available_signal_columns = [
    col
    for col in signal_columns
    if col in filtered_df.columns
]

if available_signal_columns:

    signal_data = []

    for column in available_signal_columns:

        value = pd.to_numeric(
            filtered_df[column],
            errors="coerce"
        ).fillna(0)

        signal_data.append(
            {
                "Detection Signal": column,
                "Average Score": round(
                    value.mean(),
                    4
                )
            }
        )

    signal_df = pd.DataFrame(
        signal_data
    )

    signal_fig = px.bar(
        signal_df,
        x="Detection Signal",
        y="Average Score",
        title="Detection Signal Scores"
    )

    st.plotly_chart(
        signal_fig,
        use_container_width=True
    )

else:

    st.info(
        "Detection signal data is not available."
    )


# ============================================================
# SENSOR HEALTH SUMMARY
# ============================================================

st.markdown("## Sensor Health Summary")

health_data = []

for station in sorted(
    dashboard_df["station_name"]
    .dropna()
    .unique()
):

    station_data = dashboard_df[
        dashboard_df["station_name"] == station
    ]

    records = len(station_data)

    anomalies = int(
        station_data["is_anomaly"].sum()
    )

    corrections = int(
        station_data["correction_applied"].sum()
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
            "Diagnosed Anomalies": anomalies,
            "Corrections": corrections,
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
    "Sensor health is represented as a risk signal based on observed "
    "diagnostic activity. It does not claim confirmed physical hardware failure."
)


# ============================================================
# STATION COMPARISON
# ============================================================

if len(
    dashboard_df["station_name"].unique()
) > 1:

    st.markdown("## Station Comparison")

    station_comparison = (
        dashboard_df
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
        title="Diagnosed Anomaly Rate by Station",
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
# HISTORICAL REPLAY NOTICE
# ============================================================

st.markdown("---")

st.info(
    "Demo Mode: The dashboard uses historical data replay and "
    "simulation to demonstrate anomaly monitoring. It should not "
    "be presented as a live AWS data stream."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <b>SkyGuard AI</b><br>
        Predictive Anomaly Detection, Diagnosis and Sensor Correction<br>
        P4 — Streamlit and Plotly System Integration
    </div>
    """,
    unsafe_allow_html=True
)