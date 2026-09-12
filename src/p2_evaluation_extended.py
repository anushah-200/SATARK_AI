"""
p2_evaluation_extended.py
----------------------------
SATARK AI — P2 fix: severity-level evaluation + ROC-AUC.

The original project spec asked for evaluation broken down by fault type,
severity, station, AND variable. P2's own Detector Performance Summary only
covered fault type. This fills the severity gap (fault_severity is already
in p2_day2_handoff.csv, just never grouped by it) and adds ROC-AUC, which
was also in the original spec but never reported.
"""

import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, confusion_matrix
)

DETECTOR_COLS = ["rule_score", "statistical_score", "isolation_score", "temporal_score"]


def evaluate_by_severity(results_path="person2_day2_results.csv", handoff_path="p2_day2_handoff.csv"):
    results = pd.read_csv(results_path)
    handoff = pd.read_csv(handoff_path)[["station_id", "timestamp", "fault_severity"]]
    handoff["timestamp"] = pd.to_datetime(handoff["timestamp"])
    results["timestamp"] = pd.to_datetime(results["timestamp"])

    df = results.merge(handoff, on=["station_id", "timestamp"], suffixes=("", "_truth"))
    # results already has fault_severity from itself; the merge just confirms consistency
    if "fault_severity_truth" in df.columns:
        mismatches = (df["fault_severity"] != df["fault_severity_truth"]).sum()
        print(f"fault_severity consistency check: {mismatches} mismatches between the two files (should be 0)")

    print("\n" + "=" * 60)
    print("SEVERITY-LEVEL EVALUATION (satark_prediction = anomaly_score >= 1)")
    print("=" * 60)
    df["predicted"] = (df["anomaly_score"] >= 1).astype(int)

    for severity in ["mild", "moderate", "severe"]:
        subset = df[(df["fault_severity"] == severity) | (df["is_anomaly"] == 0)]
        if subset["is_anomaly"].sum() == 0:
            continue
        p, r, f1, _ = precision_recall_fscore_support(
            subset["is_anomaly"], subset["predicted"], average="binary", zero_division=0
        )
        n = (df["fault_severity"] == severity).sum()
        print(f"{severity:10s}  n={n:3d}  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}")

    print("\n" + "=" * 60)
    print("ROC-AUC (using anomaly_score as the ranking signal)")
    print("=" * 60)
    try:
        auc = roc_auc_score(df["is_anomaly"], df["anomaly_score"])
        print(f"Overall ROC-AUC: {auc:.3f}")
        print("(Note: only 20 positives out of 43,100 rows — this number is noisy; treat as directional.)")
    except ValueError as e:
        print(f"Could not compute ROC-AUC: {e}")

    print("\n" + "=" * 60)
    print("Per-detector ROC-AUC (which individual detector ranks best?)")
    print("=" * 60)
    for col in DETECTOR_COLS:
        try:
            auc = roc_auc_score(df["is_anomaly"], df[col])
            print(f"{col:20s}  AUC={auc:.3f}")
        except ValueError:
            print(f"{col:20s}  AUC=undefined (constant column)")

    return df


if __name__ == "__main__":
    evaluate_by_severity()
