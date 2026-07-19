"""
src/monitoring.py
==================
Evidently drift monitoring — compares training data (reference)
against new incoming data (current) and generates an HTML report.

Run manually:
    python src/monitoring.py
"""
import json
from pathlib import Path

import pandas as pd
from prefect import flow
from evidently import Report
from evidently.presets import DataDriftPreset

DATA_DIR       = Path("data")
INPUT_FILE     = DATA_DIR / "features_engineered.parquet"
FEATURE_CONFIG = DATA_DIR / "feature_config.json"
REPORT_DIR     = Path("src/monitoring_reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@flow(name="aml-shield-drift-check", log_prints=True)
def run_drift_check():
    """Compare reference (older 70%) vs current (newer 30%) data slices."""
    df = pd.read_parquet(INPUT_FILE)
    with open(FEATURE_CONFIG) as f:
        feature_config = json.load(f)

    features = feature_config["all_features"]

    # Simulate "old" vs "new" data by splitting the dataset in half
    # (in real production, this would be: training data vs. live traffic)
    split_point = int(len(df) * 0.7)
    reference = df[features].iloc[:split_point]
    current   = df[features].iloc[split_point:]

    print(f"Reference (old) : {len(reference):,} rows")
    print(f"Current (new)   : {len(current):,} rows")
    print("⏳ Running drift analysis...")

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)

    output_path = REPORT_DIR / "drift_report.html"
    result.save_html(str(output_path))

    print(f"✅ Report saved: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    run_drift_check()
