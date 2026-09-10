#  SATARK AI

### Intelligent Meteorological Anomaly Detection & Sensor Health Monitoring

**SATARK AI** is an intelligent anomaly detection and sensor monitoring platform designed to identify suspicious weather observations, detect potential sensor faults, distinguish them from genuine weather events, and provide interpretable diagnostics through an interactive dashboard.

The system combines **meteorological quality-control rules, statistical detection, temporal analysis, Isolation Forest, spatial consensus, fault diagnosis, and sensor-health assessment** into a unified pipeline.

---

##  Overview

Weather stations continuously collect measurements such as:

*  Temperature
*  Humidity
*  Atmospheric Pressure

However, sensor readings can contain abnormal values caused by:

* Sudden spikes
* Gradual sensor drift
* Frozen/stuck readings
* Missing observations
* Random measurement noise
* Other sensor failures

A major challenge is that **not every unusual observation is a sensor fault**.

For example, a sudden temperature increase observed at only one station may indicate a faulty sensor, while the same increase observed across several nearby stations may represent a genuine weather event.

SATARK AI addresses this problem using multiple sources of evidence:

```text
                Weather Observation
                        │
                        ▼
              Data Preprocessing
                        │
                        ▼
              Quality Control Rules
                        │
                        ▼
       ┌────────────────────────────────┐
       │       Anomaly Detection        │
       │                                │
       │ • Statistical Detection        │
       │ • Temporal Detection           │
       │ • Isolation Forest             │
       └────────────────────────────────┘
                        │
                        ▼
              Combined SATARK Score
                        │
                        ▼
              Spatial Analysis
                        │
                        ▼
              Fault Diagnosis
                        │
                        ▼
              Sensor Health
                        │
                        ▼
             Interactive Dashboard
```

---

#  Key Features

## 1.  Meteorological Quality Control

SATARK AI applies physical plausibility checks to weather observations.

The current quality-control module checks:

| Variable    | Plausibility Range |
| ----------- | -----------------: |
| Temperature |      -50°C to 60°C |
| Humidity    |         0% to 100% |
| Pressure    |    870 to 1085 hPa |

Missing values are also flagged as suspicious.

The system generates individual flags for temperature, humidity and pressure, followed by an overall `qc_flag`.

---

## 2.  Multi-Method Anomaly Detection

SATARK AI does not rely on a single anomaly detector.

It combines three complementary signals:

### Statistical Detection

Checks whether measurements fall outside broad physical ranges.

### Temporal Detection

Uses historical observations from the same station.

For each meteorological variable, the system calculates rolling statistics and **24-hour z-scores using past observations**, reducing the risk of using future information during detection.

An observation is considered temporally anomalous when the maximum absolute z-score exceeds the configured threshold.

### Isolation Forest

A multivariate **Isolation Forest** detects unusual combinations of:

```text
Temperature
Pressure
Humidity
```

The model uses standardized numerical features and explicitly avoids ground-truth columns such as `is_anomaly`, `fault_type`, and original sensor values to prevent data leakage.

---

#  3. Synthetic Fault Injection

To evaluate the detection system, SATARK AI supports controlled injection of sensor faults into weather observations.

Currently supported fault patterns include:

###  Temperature Spike

Example:

```text
29 → 30 → 55 → 29 → 30
```

A sudden unrealistic change is injected into a measurement.

###  Sensor Drift

Example:

```text
29.0 → 30.0 → 31.0 → 32.0 → 33.0
```

The sensor gradually deviates from its original measurements.

###  Frozen / Stuck Sensor

Example:

```text
29 → 29 → 29 → 29 → 29
```

A sensor repeatedly reports the same value over consecutive observations.

###  Missing Values

A valid measurement is replaced with `NaN`.

###  Sensor Noise

Controlled random noise is added to observations using a reproducible random seed.

Each injected fault preserves the original measurement and records metadata such as:

```text
fault_type
fault_severity
is_anomaly
original_<parameter>
```

This allows the system to evaluate whether the anomaly detector successfully identifies known injected faults.

---

#  4. Spatial Anomaly Analysis

SATARK AI uses information from nearby weather stations to determine whether an abnormal observation is local or regional.

The system:

1. Calculates geographic distances between stations using the Haversine formula.
2. Identifies nearby stations within a configurable radius.
3. Retrieves measurements from neighboring stations at the same timestamp.
4. Calculates the neighboring average.
5. Measures the deviation of the target station from its neighbors.
6. Checks whether neighboring stations agree or are also anomalous.

This provides an important distinction:

```text
One station abnormal
        +
Nearby stations normal
        ↓
Potential sensor problem
```

versus

```text
Multiple nearby stations abnormal
        ↓
Potential genuine weather event
```

The spatial module currently uses station latitude/longitude and a configurable neighborhood distance.

---

#  5. Fault Diagnosis

After anomaly detection and spatial analysis, SATARK AI classifies suspicious observations using multiple pieces of evidence.

The diagnosis considers:

* ML anomaly detection
* Temporal anomaly detection
* Spatial deviation from neighboring stations

The system produces one of three classifications:

###  Sensor Fault

Strong evidence suggests that the observation is caused by a sensor problem.

###  Suspicious

The observation contains multiple warning signals but does not provide enough evidence for a definitive sensor-fault classification.

###  Likely Weather Event

The anomaly is more consistent with a genuine meteorological event.

The diagnosis module also supports value correction using the following priority:

```text
Neighbor Average
       ↓
Last Valid Value
       ↓
Original Value
```

---

#  6. SATARK Anomaly Score

The system combines detection signals into a unified anomaly score.

The resulting pipeline exposes information such as:

```text
statistical_anomaly
isolation_anomaly
temporal_anomaly
ml_anomaly_score
temporal_score
satark_score
satark_prediction
severity
reason
```

This allows downstream modules to use a consistent anomaly representation rather than relying on a single detector.

---

#  7. Interactive Dashboard

SATARK AI includes a **Streamlit + Plotly dashboard** for exploring the detected anomalies.

The dashboard provides:

### System Overview

* Total records
* Number of anomalies
* Normal observations
* Overall anomaly rate

### Latest Anomaly Alert

Displays:

* Timestamp
* Station
* Severity
* Fault type
* SATARK score
* Prediction
* Detection reason

### Spatial Diagnosis & Sensor Health

Displays:

* Diagnosis
* Sensor health score
* Neighbor average
* Spatial deviation

### Meteorological Trends

Interactive plots for:

* Temperature
* Humidity
* Pressure

Detected anomalies are highlighted directly on the time-series plots.

### Fault Distribution

Visualizes the distribution of detected fault types.

### Severity Distribution

Shows the distribution of anomaly severity levels.

### Station Comparison

Compares anomaly rates across weather stations.

### Detection Signals

Shows the contribution of:

```text
Statistical Detection
Temporal Detection
Isolation Forest
```

The dashboard also explicitly identifies the current demonstration as **historical data replay/simulation rather than a live AWS data stream**.

---

#  Project Structure

```text
SATARK_AI/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── docs/
│   └── architecture.md
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Fault_Injection.ipynb
│   ├── 04_P4_Dashboard_Integration.ipynb
│   ├── 05_P1_Day2_Quality_Control_Fault_Injection.ipynb
│   ├── P3_spatial_diagnosis_integration.ipynb
│   └── Person2_AnomalyDetection.ipynb
│
├── src/
│   ├── anomaly_detection.py
│   ├── anomaly_detector.py
│   ├── data_preprocessing.py
│   ├── diagnosis.py
│   ├── evaluate_model.py
│   ├── fault_injection.py
│   ├── person3_pipeline.py
│   ├── quality_control.py
│   ├── sensor_health.py
│   └── spatial_analysis.py
│
├── tests/
│
├── requirements.txt
├── LICENSE
└── README.md
```

The repository currently separates **data, notebooks, source modules, dashboard, tests and documentation**, making the project modular and easier to extend.

---

#  Detection Pipeline

SATARK AI follows a layered detection strategy rather than treating anomaly detection as a single classification problem.

```text
             Raw Weather Data
                    │
                    ▼
          Data Preprocessing
                    │
                    ▼
        Meteorological QC Rules
                    │
                    ▼
       ┌────────────────────────┐
       │ Multiple Detectors     │
       ├────────────────────────┤
       │ Statistical            │
       │ Temporal / Z-score     │
       │ Isolation Forest       │
       └────────────────────────┘
                    │
                    ▼
            SATARK Score
                    │
                    ▼
           Spatial Consensus
                    │
                    ▼
          Fault Classification
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Sensor    Suspicious   Weather
        Fault                  Event
          │
          ▼
       Correction
          │
          ▼
      Sensor Health
          │
          ▼
       Dashboard
```

---

#  Data Leakage Prevention

A key design consideration is preventing ground-truth information from entering the anomaly detector.

The following columns are treated as evaluation-only information:

```text
is_anomaly
fault_type
fault_severity
original_temperature
original_pressure
original_humidity
```

These values are used later to evaluate detection performance but are **not supplied as model features**.

This separation is important because otherwise the model could learn the injected fault labels directly instead of learning anomaly patterns from sensor observations.

---

#  Data Sources

SATARK AI is designed around meteorological station observations containing measurements such as:

```text
timestamp
station_id
station_name
latitude
longitude
temperature
humidity
pressure
```

The repository contains separate directories for:

```text
data/raw
data/processed
data/synthetic
```

and uses the **Meteostat** Python package as part of the project dependencies.

---

#  Tech Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Isolation Forest
* StandardScaler

### Visualization

* Matplotlib
* Plotly

### Dashboard

* Streamlit

### Model Persistence

* Joblib

### Meteorological Data

* Meteostat

These dependencies are defined in the project's `requirements.txt`.

---

#  Installation

## 1. Clone the repository

```bash
git clone https://github.com/anushah-200/SATARK_AI.git
cd SATARK_AI
```

## 2. Create a virtual environment

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

#  Running the Dashboard

From the project root:

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser.

The application expects the processed anomaly-detection results at:

```text
outputs/results/anomaly_detection_results.csv
```

The dashboard loads this result file and integrates the spatial diagnosis and sensor-health pipeline.

---

# Running the Notebooks

The notebooks provide a step-by-step view of the development pipeline.

### Exploratory Data Analysis

```text
notebooks/01_EDA.ipynb
```

### Data Preprocessing

```text
notebooks/02_Preprocessing.ipynb
```

### Fault Injection

```text
notebooks/03_Fault_Injection.ipynb
```

### Dashboard Integration

```text
notebooks/04_P4_Dashboard_Integration.ipynb
```

Additional notebooks cover quality-control fault injection, spatial diagnosis integration and anomaly detection experiments.

---

# Core Modules

| Module                  | Purpose                                        |
| ----------------------- | ---------------------------------------------- |
| `data_preprocessing.py` | Data cleaning and preparation                  |
| `quality_control.py`    | Meteorological plausibility checks             |
| `fault_injection.py`    | Synthetic sensor-fault generation              |
| `anomaly_detector.py`   | Statistical, temporal and ML anomaly detection |
| `spatial_analysis.py`   | Neighbor and spatial-consensus analysis        |
| `diagnosis.py`          | Sensor fault vs weather-event diagnosis        |
| `sensor_health.py`      | Sensor health/risk assessment                  |
| `person3_pipeline.py`   | Spatial diagnosis integration                  |
| `evaluate_model.py`     | Detection evaluation                           |
| `dashboard/app.py`      | Interactive monitoring dashboard               |

---

#  Project Objectives

SATARK AI aims to:

* Detect abnormal meteorological observations.
* Identify different sensor fault patterns.
* Combine statistical and machine-learning anomaly detection.
* Incorporate temporal context into anomaly detection.
* Use neighboring stations as contextual evidence.
* Distinguish potential sensor faults from genuine weather events.
* Provide interpretable anomaly reasons.
* Estimate sensor health based on observed anomaly activity.
* Provide an interactive monitoring interface.
* Create a foundation for future real-time weather-station monitoring.

---

#  Future Scope

SATARK AI can be extended into a real-time monitoring system with:

###  Real-Time AWS Integration

Connect directly to incoming weather-station or AWS sensor streams.

###  Streaming Anomaly Detection

Process observations continuously instead of relying on historical replay.

### Advanced ML Models

Explore:

* Autoencoders
* LSTM/GRU-based temporal models
* Temporal Transformers
* One-Class SVM
* Robust statistical models
* Ensemble anomaly detection

###  Advanced Geospatial Analysis

Integrate:

* Interactive station maps
* Weather radar
* Satellite observations
* Spatial interpolation
* Regional anomaly propagation

### Automated Alerts

Send alerts through:

* Email
* SMS
* Messaging platforms
* Monitoring systems

###  Automated Sensor Recovery

Use neighboring observations and temporal history to automatically estimate corrected sensor values.

### Long-Term Sensor Reliability

Track sensor health over time and identify stations that show persistent degradation.

---

#  Current Limitations

SATARK AI is currently designed as a **research and demonstration prototype**.

In particular:

* The dashboard currently uses historical data replay/simulation.
* It should not be interpreted as a live AWS monitoring system.
* Sensor-health scores represent anomaly-based risk and do not establish confirmed physical hardware failure.
* Detection thresholds may require calibration for different geographical regions and sensor types.
* Spatial diagnosis depends on the availability and quality of neighboring-station observations.

The dashboard itself explicitly communicates the historical-replay limitation.

---

# License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

# Contributors

**SATARK AI Team**

Built as a collaborative AI/ML project focusing on:

* Anomaly Detection
* Sensor Fault Diagnosis
* Meteorological Data Quality
* Spatial Analysis
* Machine Learning
* Data Visualization

---

#  Project Vision

SATARK AI is designed around a simple principle:

> **An unusual reading should not automatically be treated as a faulty reading.**

By combining **machine learning, temporal behavior, physical constraints, and spatial context**, SATARK AI aims to make weather-data anomaly detection more reliable, interpretable, and actionable.



