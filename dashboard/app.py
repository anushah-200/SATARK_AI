import base64
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME HELPERS
# ============================================================

def get_bg_base64(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


def style_plotly(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12, 22, 40, 0.35)",
        font=dict(color="#e8eef7", family="Segoe UI, Inter, sans-serif"),
        title_font=dict(size=16, color="#f4f7fb"),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            bgcolor="rgba(8, 16, 32, 0.45)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.12)",
        linecolor="rgba(255,255,255,0.15)",
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.12)",
        linecolor="rgba(255,255,255,0.15)",
    )
    return fig


BG_PATH = DASHBOARD_DIR / "assets" / "mountain_bg.png"
BG_DATA = get_bg_base64(BG_PATH) if BG_PATH.exists() else ""


# ============================================================
# CUSTOM STYLING — glassmorphism over mountain backdrop
# ============================================================

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Manrope', 'Segoe UI', sans-serif;
    }}

    .stApp {{
        background:
            linear-gradient(180deg, rgba(4, 12, 28, 0.55) 0%, rgba(6, 18, 38, 0.72) 100%),
            url("data:image/png;base64,{BG_DATA}");
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #eef3fa;
    }}

    [data-testid="stAppViewContainer"] > .main {{
        background: transparent;
    }}

    .main .block-container {{
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }}

    /* Sidebar glass panel */
    [data-testid="stSidebar"] {{
        background: rgba(8, 16, 32, 0.55) !important;
        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }}

    [data-testid="stSidebar"] > div:first-child {{
        background: transparent;
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {{
        color: #f2f6fc !important;
    }}

    /* Hide default Streamlit chrome */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ background: transparent !important; }}

    [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* Top action row */
    .top-actions {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.85rem;
    }}

    .icon-chip {{
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(10, 20, 40, 0.55);
        border: 1px solid rgba(255,255,255,0.12);
        color: #fff;
        font-weight: 700;
        backdrop-filter: blur(14px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }}

    .top-right {{
        display: flex;
        gap: 10px;
        align-items: center;
    }}

    .deploy-btn {{
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 55%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.65rem 1.45rem;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
        box-shadow: 0 0 0 1px rgba(99,102,241,0.35), 0 10px 28px rgba(59,130,246,0.45);
        cursor: default;
    }}

    .menu-dots {{
        width: 38px;
        height: 38px;
        border-radius: 12px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(10, 20, 40, 0.45);
        border: 1px solid rgba(255,255,255,0.1);
        color: #fff;
        font-size: 1.2rem;
        backdrop-filter: blur(12px);
    }}

    /* Hero glass banner */
    .hero-panel {{
        background: rgba(10, 18, 36, 0.48);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 1.55rem 1.75rem;
        margin-bottom: 1.35rem;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
    }}

    .hero-title {{
        font-size: 2.05rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.03em;
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }}

    .hero-link {{
        font-size: 1rem;
        opacity: 0.75;
    }}

    .hero-subtitle {{
        font-size: 1.12rem;
        font-weight: 700;
        color: #e8eef8;
        margin: 0 0 0.45rem 0;
    }}

    .hero-desc {{
        font-size: 0.95rem;
        color: rgba(226, 234, 246, 0.78);
        margin: 0;
        font-weight: 500;
    }}

    .section-kicker {{
        font-size: 0.82rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: rgba(210, 222, 240, 0.72);
        margin: 0 0 0.2rem 0;
        font-weight: 600;
    }}

    .section-title {{
        font-size: 1.55rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 1rem 0;
        letter-spacing: -0.02em;
    }}

    .glass-section {{
        background: rgba(8, 16, 32, 0.42);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.15rem 1.25rem;
        margin: 0.85rem 0 1.35rem 0;
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.28);
    }}

    .alert-card {{
        background: rgba(12, 22, 42, 0.55);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.75rem;
    }}

    .alert-badge {{
        display: inline-block;
        background: rgba(239, 68, 68, 0.18);
        color: #fecaca;
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 999px;
        padding: 0.28rem 0.75rem;
        font-weight: 700;
        font-size: 0.85rem;
        margin-bottom: 0.75rem;
    }}

    /* System Overview — rectangular glass KPI cards */
    .kpi-row {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin: 0.35rem 0 1.25rem 0;
    }}

    .kpi-card {{
        background: rgba(8, 18, 38, 0.48);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 0.95rem 1.2rem;
        min-height: 92px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28);
        box-sizing: border-box;
    }}

    .kpi-label {{
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        color: rgba(214, 226, 244, 0.78);
        margin: 0 0 0.4rem 0;
        line-height: 1.25;
    }}

    .kpi-value {{
        font-size: 1.9rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
        letter-spacing: -0.02em;
        margin: 0;
    }}

    @media (max-width: 980px) {{
        .kpi-row {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
    }}

    @media (max-width: 520px) {{
        .kpi-row {{
            grid-template-columns: 1fr;
        }}

        .kpi-value {{
            font-size: 1.65rem;
        }}
    }}

    /* Other metric glass cards (alerts, spatial, etc.) */
    [data-testid="stMetric"] {{
        background: rgba(10, 20, 40, 0.52);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.28);
    }}

    [data-testid="stMetricLabel"] {{
        color: rgba(214, 226, 244, 0.78) !important;
        font-weight: 600 !important;
    }}

    [data-testid="stMetricValue"] {{
        color: #ffffff !important;
        font-weight: 800 !important;
    }}

    /* Inputs / widgets */
    .stSelectbox label, .stTextInput label {{
        color: #e8eef8 !important;
        font-weight: 600 !important;
    }}

    div[data-baseweb="select"] > div {{
        background-color: rgba(10, 20, 40, 0.65) !important;
        border-color: rgba(255,255,255,0.14) !important;
        border-radius: 999px !important;
        color: #fff !important;
    }}

    /* Dataframes */
    [data-testid="stDataFrame"] {{
        background: rgba(8, 16, 32, 0.4);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        overflow: hidden;
    }}

    /* Alerts */
    .stAlert {{
        background: rgba(10, 20, 40, 0.55) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(12px);
    }}

    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {{
        color: #eef3fa;
    }}

    hr {{
        border-color: rgba(255,255,255,0.12) !important;
    }}

    .footer {{
        text-align: center;
        padding: 1.4rem;
        margin-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.12);
        font-size: 0.92rem;
        color: rgba(220, 230, 245, 0.78);
        background: rgba(8, 16, 32, 0.35);
        border-radius: 14px;
        backdrop-filter: blur(12px);
    }}

    /* Plotly container spacing */
    [data-testid="stPlotlyChart"] {{
        background: rgba(8, 16, 32, 0.28);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 0.35rem;
        backdrop-filter: blur(10px);
    }}
    </style>
    """,
    unsafe_allow_html=True,
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
    st.error(f"P3 result file not found: {P3_FILE}")
    st.info("Expected file: results/person3_diagnosis_correction.csv")
    st.stop()


@st.cache_data
def load_p3_data(file_path):
    data = pd.read_csv(file_path)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
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
        "correction_applied",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

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
    if not Path(file_path).exists():
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
            "wind_direction",
        ]

        available = [col for col in useful_columns if col in metadata.columns]
        return metadata[available].copy()

    except Exception:
        return pd.DataFrame()


metadata_df = load_metadata(str(METADATA_FILE))


# ============================================================
# MERGE STATION / WEATHER METADATA
# ============================================================

dashboard_df = df.copy()

if not metadata_df.empty:
    metadata_df["timestamp"] = pd.to_datetime(
        metadata_df["timestamp"],
        errors="coerce",
    )

    metadata_columns = ["timestamp", "station_id"]

    extra_columns = [
        "station_name",
        "latitude",
        "longitude",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_direction",
    ]

    available_extra = [col for col in extra_columns if col in metadata_df.columns]

    metadata_small = (
        metadata_df[metadata_columns + available_extra]
        .drop_duplicates(["timestamp", "station_id"])
    )

    dashboard_df = dashboard_df.merge(
        metadata_small,
        on=["timestamp", "station_id"],
        how="left",
    )


# ============================================================
# CREATE DASHBOARD FIELDS
# ============================================================

dashboard_df["diagnosed_fault"] = (
    dashboard_df["diagnosed_fault"].fillna("NORMAL").astype(str)
)

dashboard_df["is_anomaly"] = (
    dashboard_df["diagnosed_fault"].str.upper().ne("NORMAL")
).astype(int)

dashboard_df["correction_applied"] = (
    pd.to_numeric(dashboard_df["correction_applied"], errors="coerce")
    .fillna(0)
    .astype(int)
)

dashboard_df["diagnosis_confidence"] = pd.to_numeric(
    dashboard_df["diagnosis_confidence"],
    errors="coerce",
)

dashboard_df["anomaly_score"] = pd.to_numeric(
    dashboard_df["anomaly_score"],
    errors="coerce",
)


# ============================================================
# STATION LABEL
# ============================================================

if "station_name" not in dashboard_df.columns:
    dashboard_df["station_name"] = "Station " + dashboard_df["station_id"].astype(str)

dashboard_df["station_name"] = (
    dashboard_df["station_name"]
    .fillna("Station " + dashboard_df["station_id"].astype(str))
    .astype(str)
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## Dashboard Controls")
st.sidebar.markdown("### Station Selection")

station_options = ["All Stations"]

station_names = sorted(
    dashboard_df["station_name"].dropna().astype(str).unique().tolist()
)

station_options.extend(station_names)

selected_station = st.sidebar.selectbox(
    "Select Station",
    station_options,
)


# ============================================================
# STATION FILTER
# ============================================================

if selected_station == "All Stations":
    filtered_df = dashboard_df.copy()
else:
    filtered_df = dashboard_df[
        dashboard_df["station_name"] == selected_station
    ].copy()


if len(filtered_df) == 0:
    st.warning("No data available for the selected station.")
    st.stop()


# ============================================================
# TOP ACTIONS + HERO
# ============================================================

st.markdown(
    """
    <div class="top-actions">
        <div class="icon-chip">&gt;&gt;</div>
        <div class="top-right">
            <div class="deploy-btn">Deploy</div>
            <div class="menu-dots">⋮</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-panel">
        <div class="hero-title">SkyGuard AI <span class="hero-link">↗</span></div>
        <p class="hero-subtitle">Predictive Anomaly Detection and Sensor Monitoring</p>
        <p class="hero-desc">
            Historical anomaly detection dashboard with diagnosis, correction and sensor monitoring.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_records = len(filtered_df)
anomaly_count = int(filtered_df["is_anomaly"].sum())
normal_count = total_records - anomaly_count
correction_count = int(filtered_df["correction_applied"].sum())

if total_records > 0:
    anomaly_rate = (anomaly_count / total_records) * 100
else:
    anomaly_rate = 0


# ============================================================
# SYSTEM OVERVIEW
# ============================================================

st.markdown(
    f"""
    <p class="section-kicker">Monitoring: {selected_station}</p>
    <h2 class="section-title">System Overview</h2>
    <div class="kpi-row">
        <div class="kpi-card">
            <p class="kpi-label">Total Records</p>
            <p class="kpi-value">{total_records:,}</p>
        </div>
        <div class="kpi-card">
            <p class="kpi-label">Diagnosed Anomalies</p>
            <p class="kpi-value">{anomaly_count:,}</p>
        </div>
        <div class="kpi-card">
            <p class="kpi-label">Normal Records</p>
            <p class="kpi-value">{normal_count:,}</p>
        </div>
        <div class="kpi-card">
            <p class="kpi-label">Corrections Applied</p>
            <p class="kpi-value">{correction_count:,}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SYSTEM STATUS
# ============================================================

if anomaly_count == 0:
    st.success("System Status: Normal")
elif anomaly_rate < 5:
    st.warning(
        f"System Status: Monitoring Required — {anomaly_count:,} diagnosed anomalies"
    )
else:
    st.error(
        f"System Status: Elevated Anomaly Activity — {anomaly_count:,} diagnosed anomalies"
    )


# ============================================================
# LATEST DIAGNOSIS / ALERT
# ============================================================

st.markdown(
    """
    <div class="glass-section">
        <h2 class="section-title" style="margin-bottom:0.35rem;">
            Latest Anomaly Alert <span class="hero-link">↗</span>
        </h2>
    </div>
    """,
    unsafe_allow_html=True,
)

anomaly_df = filtered_df[filtered_df["is_anomaly"] == 1].copy()

if len(anomaly_df) > 0:
    anomaly_df = anomaly_df.sort_values("timestamp", ascending=False)
    latest = anomaly_df.iloc[0]

    alert_col1, alert_col2 = st.columns(2)

    with alert_col1:
        st.markdown(
            """
            <div class="alert-card">
                <div class="alert-badge">Anomaly Detected</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(f"**Time:** {latest['timestamp']}")
        st.write(f"**Station:** {latest['station_name']}")
        st.write(f"**Fault Type:** {latest['diagnosed_fault']}")

        confidence = pd.to_numeric(
            latest["diagnosis_confidence"],
            errors="coerce",
        )

        st.write(
            f"**Diagnosis Confidence:** {confidence:.2f}"
            if pd.notna(confidence)
            else "**Diagnosis Confidence:** N/A"
        )

    with alert_col2:
        original_temp = pd.to_numeric(latest["temperature"], errors="coerce")
        corrected_temp = pd.to_numeric(
            latest["temperature_corrected"],
            errors="coerce",
        )

        st.metric(
            "Original Temperature",
            f"{original_temp:.2f} °C" if pd.notna(original_temp) else "N/A",
        )

        st.metric(
            "Corrected Temperature",
            f"{corrected_temp:.2f} °C" if pd.notna(corrected_temp) else "N/A",
        )

        correction_method = str(latest.get("correction_method", "N/A"))
        st.write(f"**Correction Method:** {correction_method}")

        if int(latest.get("correction_applied", 0)) == 1:
            st.success("Correction Applied")
        else:
            st.info("No correction applied")

else:
    st.success("No diagnosed anomalies found for the selected station.")


# ============================================================
# DIAGNOSIS SUMMARY
# ============================================================

st.markdown("## Diagnosis Summary")

diagnosis_counts = (
    filtered_df["diagnosed_fault"].value_counts().reset_index()
)
diagnosis_counts.columns = ["diagnosed_fault", "count"]

diagnosis_fig = px.bar(
    diagnosis_counts,
    x="diagnosed_fault",
    y="count",
    title="Diagnosis Distribution",
    labels={
        "diagnosed_fault": "Diagnosis",
        "count": "Number of Records",
    },
    color_discrete_sequence=["#60a5fa"],
)
st.plotly_chart(style_plotly(diagnosis_fig), use_container_width=True)


# ============================================================
# RECENT DATA
# ============================================================

st.markdown("## Recent Sensor Data")

recent_data = (
    filtered_df.sort_values("timestamp", ascending=False).head(10)
)

display_columns = [
    "timestamp",
    "station_name",
    "temperature",
    "temperature_corrected",
    "diagnosed_fault",
    "diagnosis_confidence",
    "correction_method",
    "correction_applied",
]

available_columns = [col for col in display_columns if col in recent_data.columns]

st.dataframe(
    recent_data[available_columns],
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# TEMPERATURE MONITORING
# ============================================================

st.markdown("## Sensor Monitoring")

plot_df = filtered_df.sort_values("timestamp").copy()

if "temperature" in plot_df.columns:
    st.markdown("### Temperature Before and After Correction")

    temperature_fig = px.line(
        plot_df,
        x="timestamp",
        y=["temperature", "temperature_corrected"],
        title="Temperature Correction",
        labels={
            "timestamp": "Time",
            "value": "Temperature (°C)",
            "variable": "Series",
        },
        color_discrete_sequence=["#93c5fd", "#38bdf8"],
    )

    anomaly_points = plot_df[plot_df["is_anomaly"] == 1].copy()

    if len(anomaly_points) > 0:
        temperature_fig.add_scatter(
            x=anomaly_points["timestamp"],
            y=anomaly_points["temperature"],
            mode="markers",
            name="Diagnosed Anomaly",
            marker=dict(size=9, symbol="x", color="#f87171"),
        )

    st.plotly_chart(style_plotly(temperature_fig), use_container_width=True)


# ============================================================
# CORRECTION METHODS
# ============================================================

st.markdown("## Correction Methods")

if "correction_method" in filtered_df.columns:
    method_df = filtered_df[filtered_df["correction_applied"] == 1].copy()

    if len(method_df) > 0:
        method_counts = (
            method_df["correction_method"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .reset_index()
        )
        method_counts.columns = ["correction_method", "count"]

        method_fig = px.bar(
            method_counts,
            x="correction_method",
            y="count",
            title="Applied Correction Methods",
            labels={
                "correction_method": "Correction Method",
                "count": "Corrections Applied",
            },
            color_discrete_sequence=["#818cf8"],
        )
        st.plotly_chart(style_plotly(method_fig), use_container_width=True)
    else:
        st.info("No corrections were applied in the selected data.")


# ============================================================
# FAULT TYPE DISTRIBUTION
# ============================================================

st.markdown("## Fault Type Distribution")

fault_df = filtered_df[
    filtered_df["diagnosed_fault"].str.upper() != "NORMAL"
].copy()

if len(fault_df) > 0:
    fault_counts = fault_df["diagnosed_fault"].value_counts().reset_index()
    fault_counts.columns = ["fault_type", "count"]

    fault_fig = px.bar(
        fault_counts,
        x="fault_type",
        y="count",
        title="Diagnosed Fault Types",
        labels={
            "fault_type": "Fault Type",
            "count": "Number of Records",
        },
        color_discrete_sequence=["#38bdf8"],
    )
    st.plotly_chart(style_plotly(fault_fig), use_container_width=True)
else:
    st.info("No diagnosed fault types available.")


# ============================================================
# DIAGNOSIS CONFIDENCE
# ============================================================

st.markdown("## Diagnosis Confidence")

confidence_df = filtered_df[filtered_df["diagnosis_confidence"].notna()].copy()

if len(confidence_df) > 0:
    confidence_fig = px.histogram(
        confidence_df,
        x="diagnosis_confidence",
        nbins=20,
        title="Diagnosis Confidence Distribution",
        labels={"diagnosis_confidence": "Confidence"},
        color_discrete_sequence=["#60a5fa"],
    )
    st.plotly_chart(style_plotly(confidence_fig), use_container_width=True)
else:
    st.info("Diagnosis confidence data is not available.")


# ============================================================
# ANOMALY SCORE
# ============================================================

st.markdown("## Anomaly Score")

score_df = plot_df[plot_df["anomaly_score"].notna()].copy()

if len(score_df) > 0:
    score_fig = px.line(
        score_df,
        x="timestamp",
        y="anomaly_score",
        title="Anomaly Score Over Time",
        labels={
            "timestamp": "Time",
            "anomaly_score": "Anomaly Score",
        },
        color_discrete_sequence=["#a5b4fc"],
    )
    st.plotly_chart(style_plotly(score_fig), use_container_width=True)
else:
    st.info("Anomaly score data is not available.")


# ============================================================
# MODEL SIGNALS
# ============================================================

st.markdown("## Detection Signals")

signal_columns = [
    "rule_score",
    "statistical_score",
    "isolation_score",
    "temporal_score",
]

available_signal_columns = [
    col for col in signal_columns if col in filtered_df.columns
]

if available_signal_columns:
    signal_data = []

    for column in available_signal_columns:
        value = pd.to_numeric(filtered_df[column], errors="coerce").fillna(0)
        signal_data.append(
            {
                "Detection Signal": column,
                "Average Score": round(value.mean(), 4),
            }
        )

    signal_df = pd.DataFrame(signal_data)

    signal_fig = px.bar(
        signal_df,
        x="Detection Signal",
        y="Average Score",
        title="Detection Signal Scores",
        color_discrete_sequence=["#7dd3fc"],
    )
    st.plotly_chart(style_plotly(signal_fig), use_container_width=True)
else:
    st.info("Detection signal data is not available.")


# ============================================================
# SENSOR HEALTH SUMMARY
# ============================================================

st.markdown("## Sensor Health Summary")

health_data = []

for station in sorted(dashboard_df["station_name"].dropna().unique()):
    station_data = dashboard_df[dashboard_df["station_name"] == station]

    records = len(station_data)
    anomalies = int(station_data["is_anomaly"].sum())
    corrections = int(station_data["correction_applied"].sum())

    if records > 0:
        anomaly_percentage = (anomalies / records) * 100
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
            "Anomaly Rate (%)": round(anomaly_percentage, 2),
            "Health Status": health_status,
        }
    )

health_df = pd.DataFrame(health_data)

st.dataframe(health_df, use_container_width=True, hide_index=True)

st.caption(
    "Sensor health is represented as a risk signal based on observed "
    "diagnostic activity. It does not claim confirmed physical hardware failure."
)


# ============================================================
# STATION COMPARISON
# ============================================================

if len(dashboard_df["station_name"].unique()) > 1:
    st.markdown("## Station Comparison")

    station_comparison = (
        dashboard_df.groupby("station_name")
        .agg(
            Total_Records=("is_anomaly", "size"),
            Anomalies=("is_anomaly", "sum"),
        )
        .reset_index()
    )

    station_comparison["Anomaly_Rate"] = (
        station_comparison["Anomalies"] / station_comparison["Total_Records"]
    ) * 100

    station_comparison["Anomaly_Rate"] = station_comparison["Anomaly_Rate"].round(2)

    station_fig = px.bar(
        station_comparison,
        x="station_name",
        y="Anomaly_Rate",
        title="Diagnosed Anomaly Rate by Station",
        labels={
            "station_name": "Station",
            "Anomaly_Rate": "Anomaly Rate (%)",
        },
        color_discrete_sequence=["#60a5fa"],
    )
    st.plotly_chart(style_plotly(station_fig), use_container_width=True)


# ============================================================
# SPATIAL CONSISTENCY CHECK
# ============================================================

st.markdown("## Spatial Consistency Check")

if all(
    col in dashboard_df.columns
    for col in ["latitude", "longitude", "temperature"]
):
    spatial_station_options = sorted(
        dashboard_df["station_name"].dropna().astype(str).unique().tolist()
    )

    if len(spatial_station_options) > 1:
        spatial_station = st.selectbox(
            "Select Station",
            spatial_station_options,
            key="spatial_consistency_station",
        )

        station_data = dashboard_df[
            dashboard_df["station_name"] == spatial_station
        ].copy()

        station_data = station_data.sort_values("timestamp")

        if not station_data.empty:
            latest_station = station_data.iloc[-1]
            current_time = latest_station["timestamp"]

            comparison_df = dashboard_df[
                dashboard_df["timestamp"] == current_time
            ].copy()

            comparison_df = comparison_df.dropna(
                subset=["latitude", "longitude", "temperature"]
            )

            if len(comparison_df) > 1:
                selected_row = comparison_df[
                    comparison_df["station_name"] == spatial_station
                ]

                if not selected_row.empty:
                    selected_row = selected_row.iloc[0]

                    selected_lat = selected_row["latitude"]
                    selected_lon = selected_row["longitude"]
                    selected_temp = selected_row["temperature"]

                    comparison_df["distance"] = (
                        (comparison_df["latitude"] - selected_lat) ** 2
                        + (comparison_df["longitude"] - selected_lon) ** 2
                    ) ** 0.5

                    neighbors = (
                        comparison_df[
                            comparison_df["station_name"] != spatial_station
                        ]
                        .sort_values("distance")
                        .head(3)
                    )

                    if not neighbors.empty:
                        neighbor_average = neighbors["temperature"].mean()
                        spatial_difference = selected_temp - neighbor_average

                        c1, c2, c3 = st.columns(3)

                        with c1:
                            st.metric(
                                "Station Temperature",
                                f"{selected_temp:.2f} °C",
                            )

                        with c2:
                            st.metric(
                                "Nearby Station Average",
                                f"{neighbor_average:.2f} °C",
                            )

                        with c3:
                            st.metric(
                                "Spatial Difference",
                                f"{spatial_difference:+.2f} °C",
                            )

                        if abs(spatial_difference) >= 5:
                            st.error(
                                "⚠️ Spatial Inconsistency Detected — "
                                "the selected station differs significantly "
                                "from nearby stations."
                            )
                        else:
                            st.success(
                                "✓ Spatially Consistent — "
                                "the selected station is broadly consistent "
                                "with nearby stations."
                            )

                        spatial_plot = px.bar(
                            neighbors,
                            x="station_name",
                            y="temperature",
                            title="Nearby Station Temperature Comparison",
                            labels={
                                "station_name": "Station",
                                "temperature": "Temperature (°C)",
                            },
                            color_discrete_sequence=["#38bdf8"],
                        )

                        spatial_plot.add_hline(
                            y=selected_temp,
                            line_dash="dash",
                            annotation_text=(
                                f"{spatial_station}: {selected_temp:.2f} °C"
                            ),
                        )

                        st.plotly_chart(
                            style_plotly(spatial_plot),
                            use_container_width=True,
                        )

                        st.caption(
                            "Spatial consistency is a risk signal based on "
                            "nearby station observations. It helps distinguish "
                            "station-specific anomalies from broader weather variation."
                        )
                    else:
                        st.info(
                            "No nearby station data available for comparison."
                        )
                else:
                    st.info(
                        "Selected station data is unavailable at the latest timestamp."
                    )
            else:
                st.info(
                    "Not enough simultaneous station observations "
                    "for spatial comparison."
                )
    else:
        st.info("Spatial consistency requires data from multiple stations.")
else:
    st.warning(
        "Latitude, longitude, or temperature data is unavailable. "
        "Spatial consistency check cannot be performed."
    )


# ============================================================
# HISTORICAL REPLAY NOTICE
# ============================================================

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
    unsafe_allow_html=True,
)
