"""
feature_engineering_job.py
===========================
Runs as a SageMaker Processing Job. Reads raw HI-Small_Trans.csv from
the mounted input directory, applies the same 5 feature engineering
functions used throughout this project, filters down to the exact
66-feature set, and writes both the parquet output and the feature
config JSON — matching 02_feature_engineering.ipynb exactly.

SageMaker automatically mounts:
    S3 input  -> /opt/ml/processing/input/
    S3 output -> /opt/ml/processing/output/  (auto-uploaded back to S3
                 after the script finishes)
"""
import json
import gc
import numpy as np
import pandas as pd

INPUT_PATH = "/opt/ml/processing/input/HI-Small_Trans.csv"
OUTPUT_PARQUET_PATH = "/opt/ml/processing/output/features_engineered.parquet"
OUTPUT_CONFIG_PATH = "/opt/ml/processing/output/feature_config.json"


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['hour']            = out['Timestamp'].dt.hour
    out['day_of_week']     = out['Timestamp'].dt.dayofweek
    out['is_weekend']      = (out['day_of_week'] >= 5).astype(int)
    out['is_night']        = out['hour'].apply(
        lambda h: 1 if (h >= 22 or h <= 6) else 0
    )
    out['is_business_hrs'] = out.apply(
        lambda r: 1 if (9 <= r['hour'] <= 17 and r['day_of_week'] < 5) else 0,
        axis=1
    )
    return out


def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['log_amount_paid']     = np.log1p(out['Amount Paid'])
    out['log_amount_received'] = np.log1p(out['Amount Received'])
    out['amount_difference'] = np.abs(out['Amount Paid'] - out['Amount Received'])
    out['amount_ratio']      = out['Amount Paid'] / (out['Amount Received'] + 1e-6)
    out['is_round_100']   = (out['Amount Paid'] % 100   == 0).astype(int)
    out['is_round_1000']  = (out['Amount Paid'] % 1000  == 0).astype(int)
    out['is_round_10000'] = (out['Amount Paid'] % 10000 == 0).astype(int)
    out['amount_band'] = pd.cut(
        out['Amount Paid'],
        bins=[0, 100, 1_000, 10_000, 100_000, np.inf],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True
    ).astype(int)
    return out


def create_network_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['is_same_bank']      = (out['From Bank'] == out['To Bank']).astype(int)
    out['is_cross_currency'] = (out['Payment Currency'] != out['Receiving Currency']).astype(int)
    out['is_self_loop']      = (out['From Account'] == out['To Account']).astype(int)
    return out


def create_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sender_agg = (
        out.groupby('From Account')
        .agg(
            sender_tx_count    =('Amount Paid', 'count'),
            sender_avg_amount  =('Amount Paid', 'mean'),
            sender_std_amount  =('Amount Paid', 'std'),
            sender_max_amount  =('Amount Paid', 'max'),
            sender_unique_banks=('To Bank', 'nunique'),
        )
        .reset_index()
    )
    sender_agg['sender_std_amount'] = sender_agg['sender_std_amount'].fillna(0)

    receiver_agg = (
        out.groupby('To Account')
        .agg(
            receiver_tx_count  =('Amount Received', 'count'),
            receiver_avg_amount=('Amount Received', 'mean'),
        )
        .reset_index()
    )

    out = out.merge(sender_agg,   on='From Account', how='left')
    out = out.merge(receiver_agg, on='To Account',   how='left')

    out['amount_vs_sender_avg'] = (
        out['Amount Paid'] / (out['sender_avg_amount'] + 1e-6)
    )
    out['sender_amount_zscore'] = (
        (out['Amount Paid'] - out['sender_avg_amount'])
        / (out['sender_std_amount'] + 1e-6)
    ).clip(-10, 10)
    return out


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, prefix in [
        ('Payment Format',     'fmt'),
        ('Receiving Currency', 'recv_ccy'),
        ('Payment Currency',   'pay_ccy'),
    ]:
        dummies = pd.get_dummies(out[col], prefix=prefix, dtype=int)
        out = pd.concat([out, dummies], axis=1)

    for col, new_col in [
        ('From Bank', 'from_bank_freq'),
        ('To Bank',   'to_bank_freq'),
    ]:
        freq_map = out[col].value_counts(normalize=True).to_dict()
        out[new_col] = out[col].map(freq_map)

    return out


if __name__ == "__main__":
    print(f"Reading raw data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    for c in df.select_dtypes(include="float64").columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include="int64").columns:
        df[c] = df[c].astype("int32")
    for c in df.select_dtypes(include="float64").columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include="int64").columns:
        df[c] = df[c].astype("int32")
    df = df.rename(columns={"Account": "From Account", "Account.1": "To Account"})
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M')
    df['transaction_id'] = df.index

    print("Applying feature engineering pipeline...")
    df = create_temporal_features(df)
    gc.collect()
    df = create_amount_features(df)
    gc.collect()
    df = create_network_features(df)
    gc.collect()
    df = create_velocity_features(df)
    gc.collect()
    df = encode_categorical_features(df)
    gc.collect()

    # ── Define final feature groups — matches 02_feature_engineering.ipynb ──
    TEMPORAL_FEATURES = ['hour', 'day_of_week', 'is_weekend', 'is_night', 'is_business_hrs']
    AMOUNT_FEATURES = [
        'Amount Paid', 'Amount Received',
        'log_amount_paid', 'log_amount_received',
        'amount_difference', 'amount_ratio',
        'is_round_100', 'is_round_1000', 'is_round_10000',
        'amount_band',
    ]
    NETWORK_FEATURES = ['is_same_bank', 'is_cross_currency', 'is_self_loop']
    VELOCITY_FEATURES = [
        'sender_tx_count', 'sender_avg_amount', 'sender_std_amount',
        'sender_max_amount', 'sender_unique_banks',
        'receiver_tx_count', 'receiver_avg_amount',
        'amount_vs_sender_avg', 'sender_amount_zscore',
    ]
    ENCODED_FEATURES = (
        [c for c in df.columns if c.startswith('fmt_')]
        + [c for c in df.columns if c.startswith('recv_ccy_')]
        + [c for c in df.columns if c.startswith('pay_ccy_')]
        + ['from_bank_freq', 'to_bank_freq']
    )
    TARGET = 'Is Laundering'
    ALL_FEATURES = (
        TEMPORAL_FEATURES + AMOUNT_FEATURES + NETWORK_FEATURES
        + VELOCITY_FEATURES + ENCODED_FEATURES
    )

    missing_features = [f for f in ALL_FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # ── Select final columns, matching 02_feature_engineering.ipynb ──
    df_model = df[ALL_FEATURES + [TARGET] + ["transaction_id"]].copy()

    null_counts = df_model.isnull().sum()
    if null_counts.sum() > 0:
        print(f"Nulls found — filling with 0:\n{null_counts[null_counts > 0]}")
        df_model = df_model.fillna(0)
    else:
        print("No null values in final feature set")

    print(f"Final shape: {df_model.shape}")
    print(f"Writing parquet to {OUTPUT_PARQUET_PATH}...")
    df_model.to_parquet(OUTPUT_PARQUET_PATH, index=False)

    feature_config = {
        "all_features": ALL_FEATURES,
        "temporal_features": TEMPORAL_FEATURES,
        "amount_features": AMOUNT_FEATURES,
        "network_features": NETWORK_FEATURES,
        "velocity_features": VELOCITY_FEATURES,
        "encoded_features": ENCODED_FEATURES,
        "target": TARGET,
        "total_features": len(ALL_FEATURES),
    }
    print(f"Writing feature config to {OUTPUT_CONFIG_PATH}...")
    with open(OUTPUT_CONFIG_PATH, "w") as f:
        json.dump(feature_config, f, indent=2)

    print("Done.")
