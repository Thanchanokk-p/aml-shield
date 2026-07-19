"""
src/fairness_audit.py
======================
Checks whether the model performs equally well across different
payment format groups (fairness/bias check).

Run manually:
    python src/fairness_audit.py
"""
import json
from pathlib import Path

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR       = Path("data")
INPUT_FILE     = DATA_DIR / "features_engineered.parquet"
FEATURE_CONFIG = DATA_DIR / "feature_config.json"

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


def run_fairness_audit():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("aml-shield-experiments")

    df = pd.read_parquet(INPUT_FILE)
    with open(FEATURE_CONFIG) as f:
        feature_config = json.load(f)

    features = feature_config["all_features"]
    target   = feature_config["target"]

    X = df[features]
    y = df[target]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Load the latest trained model (native XGBoost format)
    latest_run = mlflow.search_runs(
        order_by=["start_time DESC"], max_results=1
    )
    run_id = latest_run.iloc[0]["run_id"]
    model_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="xgboost_model/model.ubj"
    )
    model = xgb.Booster()
    model.load_model(model_path)

    dtest = xgb.DMatrix(X_test)
    probs = model.predict(dtest)

    # payment format columns are one-hot encoded (fmt_ACH, fmt_Wire, ...)
    fmt_cols = [c for c in features if c.startswith("fmt_")]

    print("=" * 65)
    print("FAIRNESS AUDIT — Performance by Payment Format")
    print("=" * 65)

    results = []
    for col in fmt_cols:
        mask = X_test[col] == 1
        n = mask.sum()
        if n < 50:
            continue
        try:
            auc = roc_auc_score(y_test[mask], probs[mask])
        except ValueError:
            auc = None
        fmt_name = col.replace("fmt_", "")
        results.append({"format": fmt_name, "n": int(n), "auc": auc})
        auc_str = f"{auc:.4f}" if auc is not None else "N/A (only 1 class)"
        print(f"  {fmt_name:<15} n={n:>8,}   AUC={auc_str}")

    print()
    aucs = [r["auc"] for r in results if r["auc"] is not None]
    if aucs:
        spread = max(aucs) - min(aucs)
        print(f"AUC spread across formats: {spread:.4f}")
        if spread > 0.10:
            print("⚠️  Meaningful performance gap across payment formats.")
        else:
            print("✅ Performance is reasonably consistent across formats.")

    return results


if __name__ == "__main__":
    run_fairness_audit()
