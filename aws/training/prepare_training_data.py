"""
prepare_training_data.py
=========================
Converts features_engineered.parquet into the CSV format SageMaker's
built-in XGBoost algorithm requires:
    - target column first, no header, no index
Splits into train/validation using the same 70/15 split and
random_state=42 as 03_baseline_model_mlflow.ipynb, so results stay
comparable to the local model.

Run this on your Mac — it's a light local step, not a SageMaker job.
"""
import json

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "../../data"
FEATURES_PATH = f"{DATA_DIR}/features_engineered.parquet"
CONFIG_PATH = f"{DATA_DIR}/feature_config.json"

OUTPUT_TRAIN_CSV = "train.csv"
OUTPUT_VAL_CSV = "validation.csv"
OUTPUT_SCALE_POS_WEIGHT = "scale_pos_weight.txt"

print("Loading features_engineered.parquet...")
df = pd.read_parquet(FEATURES_PATH)

with open(CONFIG_PATH) as f:
    config = json.load(f)

ALL_FEATURES = config["all_features"]
TARGET = config["target"]

X = df[ALL_FEATURES]
y = df[TARGET]

X_temp, _, y_temp, _ = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=(0.15 / 0.85), random_state=42, stratify=y_temp
)

train_df = pd.concat([y_train, X_train], axis=1)
val_df = pd.concat([y_val, X_val], axis=1)

train_df.to_csv(OUTPUT_TRAIN_CSV, header=False, index=False)
val_df.to_csv(OUTPUT_VAL_CSV, header=False, index=False)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
with open(OUTPUT_SCALE_POS_WEIGHT, "w") as f:
    f.write(str(scale_pos_weight))

print(f"Train shape: {train_df.shape} -> {OUTPUT_TRAIN_CSV}")
print(f"Validation shape: {val_df.shape} -> {OUTPUT_VAL_CSV}")
print(f"scale_pos_weight: {scale_pos_weight:.2f} -> {OUTPUT_SCALE_POS_WEIGHT}")
print("Done.")
