"""
src/pipeline.py
================
Prefect flow — automates the core AML-Shield training pipeline.
Converted from notebooks/03_baseline_model_mlflow.ipynb (sections 0-3 only).

Excludes: SHAP, visualization, business cost model, ACH dependency,
speaker docs — those stay in the notebook for manual/exploratory use.

Run manually:
    python src/pipeline.py

Run via Prefect CLI (after deploying):
    prefect deployment run 'training-pipeline/aml-shield'
"""
import json
import warnings
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from prefect import flow, task
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths (relative to project root, not notebooks/) ───────────────
DATA_DIR       = Path("data")
INPUT_FILE     = DATA_DIR / "features_engineered.parquet"
FEATURE_CONFIG = DATA_DIR / "feature_config.json"

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME     = "aml-shield-experiments"


@task(name="load-data", retries=2, retry_delay_seconds=10)
def load_data():
    """Load engineered features + target from parquet, per feature_config.json."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_parquet(INPUT_FILE)
    with open(FEATURE_CONFIG) as f:
        feature_config = json.load(f)

    all_features = feature_config["all_features"]
    target       = feature_config["target"]

    X = df[all_features]
    y = df[target]

    print(f"✅ Loaded {len(df):,} rows, {X.shape[1]} features, "
          f"fraud={y.mean()*100:.2f}%")
    return X, y


@task(name="split-data")
def split_data(X: pd.DataFrame, y: pd.Series):
    """3-way stratified split: 70% train / 15% val / 15% test."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=(0.15 / 0.85), random_state=42, stratify=y_temp
    )
    print(f"✅ Split — train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def _evaluate(model, X, y, threshold=0.5, name="dataset"):
    """Shared evaluation helper (not a task — called inside train_and_log)."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        f"{name}_auc_roc":       round(roc_auc_score(y, y_prob), 4),
        f"{name}_avg_precision": round(average_precision_score(y, y_prob), 4),
        f"{name}_f1_fraud":      round(f1_score(y, y_pred, pos_label=1, zero_division=0), 4),
        f"{name}_recall_fraud":  round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
        f"{name}_precision_fraud": round(tp / (tp + fp) if (tp + fp) > 0 else 0, 4),
        f"{name}_true_positives":  int(tp),
        f"{name}_false_positives": int(fp),
        f"{name}_false_negatives": int(fn),
        f"{name}_true_negatives":  int(tn),
    }
    return metrics, y_prob


@task(name="train-and-log-model", retries=1)
def train_and_log_model(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train XGBoost, evaluate on val+test, log everything to MLflow."""
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    params = {
        "n_estimators": 300, "max_depth": 6, "min_child_weight": 5,
        "subsample": 0.8, "colsample_bytree": 0.8, "learning_rate": 0.05,
        "gamma": 1, "scale_pos_weight": imbalance_ratio,
        "random_state": 42, "n_jobs": -1, "eval_metric": "aucpr",
        "tree_method": "hist", "verbosity": 0,
    }

    with mlflow.start_run(run_name="xgboost_prefect_flow") as run:
        run_id = run.info.run_id
        mlflow.log_params(params)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("imbalance_ratio", round(imbalance_ratio, 2))

        print("⏳ Training XGBoost...")
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print("✅ Training complete")

        val_metrics, _   = _evaluate(model, X_val, y_val, name="val")
        test_metrics, _  = _evaluate(model, X_test, y_test, name="test")
        mlflow.log_metrics({**val_metrics, **test_metrics})

        # Native XGBoost save — bypasses sklearn mixin compatibility issues
        # (same fix proven to work on AWS EC2 deployment, Phase 2)
        model_path = "model.ubj"
        model.get_booster().save_model(model_path)
        mlflow.log_artifact(model_path, artifact_path="xgboost_model")
        import os
        os.remove(model_path)

        print(f"✅ Run logged — ID: {run_id}")
        print(f"   Test AUC-ROC: {test_metrics['test_auc_roc']}")
        print(f"   Test Recall (Fraud): {test_metrics['test_recall_fraud']}")

    return run_id, test_metrics


@flow(name="aml-shield-training-pipeline", log_prints=True)
def training_pipeline():
    """Full training pipeline — load → split → train → log."""
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    run_id, test_metrics = train_and_log_model(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    print(f"\n🎉 Pipeline complete — MLflow run: {run_id}")
    return run_id, test_metrics


if __name__ == "__main__":
    training_pipeline()
