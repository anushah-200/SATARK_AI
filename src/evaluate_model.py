"""
evaluate_model.py
------------------
SATARK AI — P2 secondary deliverable: anomaly-score evaluation.

Run against synthetic_fault_data.csv, which has ground-truth is_anomaly /
fault_type / fault_severity columns (these are NEVER fed into the model).
"""

import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)
from anomaly_detector import run_anomaly_pipeline


def evaluate(df: pd.DataFrame, pred_col="satark_prediction",
             score_col="satark_score", truth_col="is_anomaly"):
    df = df.dropna(subset=[pred_col, score_col])

    print("=" * 60)
    print("OVERALL PERFORMANCE")
    print("=" * 60)
    print(classification_report(df[truth_col], df[pred_col]))
    print("Confusion matrix:\n", confusion_matrix(df[truth_col], df[pred_col]))
    try:
        auc = roc_auc_score(df[truth_col], df[score_col])
        print(f"ROC-AUC: {auc:.3f}")
    except ValueError:
        print("ROC-AUC: not computable (only one class present)")

    print("\n" + "=" * 60)
    print("PERFORMANCE BY FAULT TYPE")
    print("=" * 60)
    for fault in df["fault_type"].unique():
        if fault == "normal":
            continue
        subset = df[(df["fault_type"] == fault) | (df[truth_col] == 0)]
        p, r, f1, _ = precision_recall_fscore_support(
            subset[truth_col], subset[pred_col], average="binary", zero_division=0
        )
        print(f"{fault:30s}  precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}")

    print("\n" + "=" * 60)
    print("PERFORMANCE BY SEVERITY")
    print("=" * 60)
    for sev in ["mild", "moderate", "severe"]:
        subset = df[(df["fault_severity"] == sev) | (df[truth_col] == 0)]
        if subset[truth_col].sum() == 0:
            continue
        p, r, f1, _ = precision_recall_fscore_support(
            subset[truth_col], subset[pred_col], average="binary", zero_division=0
        )
        print(f"{sev:10s}  precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}")

    print("\n" + "=" * 60)
    print("PERFORMANCE BY STATION")
    print("=" * 60)
    for station in df["station_name"].unique():
        subset = df[df["station_name"] == station]
        if subset[truth_col].sum() == 0:
            continue
        p, r, f1, _ = precision_recall_fscore_support(
            subset[truth_col], subset[pred_col], average="binary", zero_division=0
        )
        print(f"{station:15s}  precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "synthetic_fault_data.csv"
    data = pd.read_csv(path)
    result = run_anomaly_pipeline(data)
    result.to_csv("anomaly_detection_results.csv", index=False)
    evaluate(result)
