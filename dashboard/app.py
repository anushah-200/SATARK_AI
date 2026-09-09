import os

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# TITLE
# ============================================================

st.title("🛰️ SkyGuard AI")
st.subheader("Predictive Anomaly Detection & Sensor Monitoring")

st.caption(
    "Historical anomaly detection dashboard with replay/simulation-based monitoring."
)


# ============================================================
# LOAD DATA
# ============================================================

RESULT_FILE = "outputs/results/anomaly_detection_results.csv"

if not os.path.exists(RESULT_FILE):
    st.error(
        f"Result file not found: {RESULT_FILE}"
    )
    st.stop()

try:
    df = pd.read_csv(RESULT_FILE)
except Exception as e:
    st.error(f"Unable to load result file: {e}")
    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

# Convert anomaly column safely
df["is_anomaly"] = pd.to_numeric(
    df["is_anomaly"],
    errors="coerce"
).fillna(0)

# Make a copy for dashboard use
dashboard_df = df.copy()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎛️ Dashboard Controls")

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


# Apply station filter
if selected_station == "All Stations":
    filtered_df = dashboard_df.copy()
else:
    filtered_df = dashboard_df[
        dashboard_df["station_name"].astype(str)
        == selected_station
    ].copy()


# ============================================================
# HEADER STATUS
# ============================================================

st.markdown("---")

if len(filtered_df) == 0:
    st.warning("No data available for the selected station.")
    st.stop()


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_records = len(filtered_df)

anomaly_count = int(
    filtered_df["is_anomaly"].sum()
)

normal_count = total_records - anomaly_count

if total_records > 0:
    anomaly_rate = (
        anomaly_count / total_records
    ) * 100
else:
    anomaly_rate = 0


# Number of stations
if "station_name" in filtered_df.columns:
    station_count = (
        filtered_df["station_name"]
        .nunique()
    )
else:
    station_count = 0


# ============================================================
# KPI SECTION
# ============================================================

st.markdown("## 📊 System Overview")

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
        "🟢 System Status: Normal"
    )
elif anomaly_rate < 5:
    st.warning(
        f"🟡 System Status: Monitoring Required — "
        f"{anomaly_count:,} anomalies detected"
    )
else:
    st.error(
        f"🔴 System Status: Elevated Anomaly Activity — "
        f"{anomaly_count:,} anomalies detected"
    )


# ============================================================
# LATEST ANOMALY ALERT
# ============================================================

st.markdown("## 🚨 Latest Anomaly Alert")

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
            "⚠️ Anomaly Detected"
        )

        st.write(
            f"**Time:** {latest['timestamp']}"
        )

        if "station_name" in latest:
            st.write(
                f"**Station:** {latest['station_name']}"
            )

        if "severity" in latest:
            st.write(
                f"**Severity:** {latest['severity']}"
            )

        if "fault_type" in latest:
            st.write(
                f"**Fault Type:** {latest['fault_type']}"
            )

    with alert_col2:

        if "satark_score" in latest:
            st.metric(
                "Satark Score",
                f"{latest['satark_score']:.3f}"
                if pd.notna(latest["satark_score"])
                else "N/A"
            )

        if "satark_prediction" in latest:
            st.write(
                f"**Prediction:** "
                f"{latest['satark_prediction']}"
            )

        if "reason" in latest:
            reason = str(latest["reason"])

            if reason.lower() != "nan":
                st.info(
                    f"**Reason:** {reason}"
                )

else:

    st.success(
        "✅ No anomalies found for the selected station."
    )


# ============================================================
# RECENT DATA TABLE
# ============================================================

st.markdown("## 📋 Recent Sensor Data")

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
    col for col in display_columns
    if col in recent_data.columns
]

st.dataframe(
    recent_data[available_columns],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SENSOR MONITORING
# ============================================================

st.markdown("## 📈 Sensor Monitoring")

plot_df = filtered_df.sort_values(
    "timestamp"
).copy()

plot_anomalies = plot_df[
    plot_df["is_anomaly"] == 1
]


# ============================================================
# TEMPERATURE GRAPH
# ============================================================

if "temperature" in plot_df.columns:

    st.markdown("### 🌡️ Temperature")

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

        temperature_anomalies = plot_anomalies[
            ["timestamp", "temperature"]
        ].dropna()

        temperature_fig.add_scatter(
            x=temperature_anomalies["timestamp"],
            y=temperature_anomalies["temperature"],
            mode="markers",
            name="Detected Anomaly",
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
# HUMIDITY GRAPH
# ============================================================

if "humidity" in plot_df.columns:

    st.markdown("### 💧 Humidity")

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

        humidity_anomalies = plot_anomalies[
            ["timestamp", "humidity"]
        ].dropna()

        humidity_fig.add_scatter(
            x=humidity_anomalies["timestamp"],
            y=humidity_anomalies["humidity"],
            mode="markers",
            name="Detected Anomaly",
            marker=dict(
                size=9,
                symbol="x"
            )
        )

    st.plotly_chart(
        humidity_fig,
        use_container_width=True
    )


# ============================================================
# PRESSURE GRAPH
# ============================================================

if "pressure" in plot_df.columns:

    st.markdown("### 🌬️ Pressure")

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

        pressure_anomalies = plot_anomalies[
            ["timestamp", "pressure"]
        ].dropna()

        pressure_fig.add_scatter(
            x=pressure_anomalies["timestamp"],
            y=pressure_anomalies["pressure"],
            mode="markers",
            name="Detected Anomaly",
            marker=dict(
                size=9,
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

    st.markdown("## 🧩 Fault Type Distribution")

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

    st.markdown("## ⚠️ Severity Distribution")

    severity_df = filtered_df[
        filtered_df["is_anomaly"] == 1
    ].copy()

    if len(severity_df) > 0:

        severity_counts = (
            severity_df[severity_column]
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
# SENSOR HEALTH
# ============================================================

st.markdown("## 🩺 Sensor Health")

if "station_name" in filtered_df.columns:

    health_data = []

    for station in filtered_df[
        "station_name"
    ].dropna().unique():

        station_data = filtered_df[
            filtered_df["station_name"] == station
        ]

        records = len(station_data)

        anomalies = int(
            station_data["is_anomaly"].sum()
        )

        if records > 0:
            anomaly_percentage = (
                anomalies / records
            ) * 100
        else:
            anomaly_percentage = 0

        # Risk-based health interpretation
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
                    round(anomaly_percentage, 2),
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
        "observed anomaly activity; it does not claim confirmed "
        "physical hardware failure."
    )


# ============================================================
# STATION COMPARISON
# ============================================================

if "station_name" in filtered_df.columns:

    st.markdown("## 🏭 Station Comparison")

    station_comparison = (
        filtered_df
        .groupby("station_name")
        .agg(
            Total_Records=("is_anomaly", "size"),
            Anomalies=("is_anomaly", "sum")
        )
        .reset_index()
    )

    station_comparison[
        "Anomaly_Rate"
    ] = (
        station_comparison["Anomalies"]
        / station_comparison["Total_Records"]
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

    st.markdown("## 🎯 Satark Anomaly Score")

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

st.markdown("## 🤖 Detection Signals")

signal_columns = [
    "statistical_anomaly",
    "temporal_anomaly",
    "isolation_anomaly"
]

available_signal_columns = [
    col for col in signal_columns
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
                list(signal_counts.keys()),
            "Detected Records":
                list(signal_counts.values())
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
    "🔄 Demo Mode: The current dashboard uses historical data "
    "replay/simulation to demonstrate anomaly monitoring. "
    "It should not be presented as a live AWS data stream."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="text-align:center; padding:20px;">
        <b>SkyGuard AI</b><br>
        Predictive Anomaly Detection & Sensor Monitoring<br>
        <small>P4 — Streamlit & Plotly System Integration</small>
    </div>
    """,
    unsafe_allow_html=True
)