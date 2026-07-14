"""
src/features.py
================
Converts a raw transaction dictionary into the 66-feature vector
that the XGBoost model was trained on.

Why this file exists:
    The feature engineering logic lives in notebook 02.
    Notebooks cannot be imported by other Python files.
    So we copy the logic here as plain functions that the
    API (api.py) can call on every request.

How it is used:
    from src.features import build_feature_vector
    features = build_feature_vector(transaction_dict)
    risk_score = model.predict_proba(features)[0][1]
"""

# ── Imports ────────────────────────────────────────────────────────

import json                 # reads the feature_config.json file
import numpy as np          # math operations (log, abs, etc.)
import pandas as pd         # DataFrame — the format model expects
from pathlib import Path    # handles file paths across Mac/Windows


# ── Constants ──────────────────────────────────────────────────────
# These must match EXACTLY what was used during training.
# If a payment format is missing or misspelled, the one-hot encoding
# will be wrong and the model will silently produce bad predictions.

ALL_PAYMENT_FORMATS = [
    "ACH",
    "Bitcoin",
    "Cash",
    "Cheque",
    "Credit Card",
    "Reinvestment",
    "Wire",
]

ALL_CURRENCIES = [
    "Australian Dollar",
    "Bitcoin",
    "Brazil Real",
    "Canadian Dollar",
    "Euro",
    "Mexican Peso",
    "Ruble",
    "Rupee",
    "Saudi Riyal",
    "Shekel",
    "Swiss Franc",
    "UK Pound",
    "US Dollar",
    "Yen",
    "Yuan",
]


# ── Feature group functions ────────────────────────────────────────
# Each function builds one group of features.
# Splitting into functions makes the code easier to test and read.


def _build_temporal_features(hour: int, day_of_week: int) -> dict:
    """
    Build time-based features from hour and day of week.

    hour: int        0 to 23       e.g. 2 means 2am
    day_of_week: int 0 to 6        0=Monday, 6=Sunday

    Returns a dict like:
        {"hour": 2, "is_night": 1, "is_weekend": 1, ...}
    """

    # int(...) converts True/False to 1/0
    # ML models need numbers not booleans

    is_weekend = int(day_of_week >= 5)
    # >= 5 means Saturday(5) or Sunday(6)

    is_night = int(hour >= 22 or hour <= 6)
    # 10pm to 6am = off-hours = slightly more suspicious

    is_business_hrs = int(9 <= hour <= 17 and day_of_week < 5)
    # 9am to 5pm on a weekday = normal business hours

    return {
        "hour"            : hour,
        "day_of_week"     : day_of_week,
        "is_weekend"      : is_weekend,
        "is_night"        : is_night,
        "is_business_hrs" : is_business_hrs,
    }


def _build_amount_features(amount_paid: float, amount_received: float) -> dict:
    """
    Build features from transaction amounts.

    Why log transform?
        Raw amounts range from 0.01 to 1,000,000,000+
        That range is too large for the model to learn from.
        log(x+1) compresses it:
            1        -> 0.69
            1,000    -> 6.91
            1,000,000 -> 13.8
        Every time the amount multiplies by 10,
        the log value increases by ~2.3 which is manageable.

    Why +1 in log1p?
        log(0) = undefined (error)
        log(0+1) = log(1) = 0 which is safe
        numpy.log1p(x) = log(x+1) built-in
    """

    log_paid     = np.log1p(amount_paid)
    log_received = np.log1p(amount_received)

    # Difference between sent and received
    # Large gap may indicate currency conversion fees used in layering
    difference = abs(amount_paid - amount_received)

    # Ratio of sent to received
    # + 1e-6 prevents division by zero (1e-6 = 0.000001)
    ratio = amount_paid / (amount_received + 1e-6)

    # Round number flags
    # Structuring = breaking large amounts into round numbers to avoid detection
    # % = modulo operator: 50000 % 1000 = 0 (exactly divisible)
    is_round_100   = int(amount_paid % 100   == 0)
    is_round_1000  = int(amount_paid % 1000  == 0)
    is_round_10000 = int(amount_paid % 10000 == 0)

    # Amount band = size bucket
    # 0=micro, 1=small, 2=medium, 3=large, 4=whale
    if   amount_paid <= 100:      band = 0
    elif amount_paid <= 1_000:    band = 1
    elif amount_paid <= 10_000:   band = 2
    elif amount_paid <= 100_000:  band = 3
    else:                         band = 4

    return {
        "Amount Paid"         : amount_paid,
        "Amount Received"     : amount_received,
        "log_amount_paid"     : log_paid,
        "log_amount_received" : log_received,
        "amount_difference"   : difference,
        "amount_ratio"        : ratio,
        "is_round_100"        : is_round_100,
        "is_round_1000"       : is_round_1000,
        "is_round_10000"      : is_round_10000,
        "amount_band"         : band,
    }


def _build_network_features(is_same_bank: int, is_cross_currency: int) -> dict:
    """
    Build features about the transaction network relationship.

    is_same_bank:      1 if both accounts are at the same bank
    is_cross_currency: 1 if payment and receiving currencies differ

    Note on is_self_loop:
        Would be 1 if sender and receiver are the same account.
        We cannot know this at API time so we default to 0.
        This is a known production limitation documented in FAILURES.md.
    """
    return {
        "is_same_bank"      : is_same_bank,
        "is_cross_currency" : is_cross_currency,
        "is_self_loop"      : 0,
    }


def _build_velocity_features(
    amount_paid: float,
    sender_tx_count: int,
    sender_avg_amount: float,
) -> dict:
    """
    Build velocity features — how fast is this account moving money?

    Interesting finding from EDA:
        Accounts with 1-5 transactions had the HIGHEST fraud rate (0.29%).
        Expected high-velocity accounts to be most suspicious.
        The opposite was true — mule accounts transact once then go dormant.

    amount_vs_sender_avg:
        How unusual is this transaction vs account history?
        amount_paid=50000 but sender_avg=500 means ratio=100
        100x above normal = very suspicious
    """
    amount_vs_avg = amount_paid / (sender_avg_amount + 1e-6)

    return {
        "sender_tx_count"     : sender_tx_count,
        "sender_avg_amount"   : sender_avg_amount,
        "sender_std_amount"   : 0,
        "sender_max_amount"   : amount_paid,
        "sender_unique_banks" : 1,
        "receiver_tx_count"   : 1,
        "receiver_avg_amount" : amount_paid,
        "amount_vs_sender_avg": amount_vs_avg,
        "sender_amount_zscore": 0,
        "from_bank_freq"      : 0.001,
        "to_bank_freq"        : 0.001,
    }


def _build_encoded_features(payment_format: str) -> dict:
    """
    One-hot encode the payment format and currencies.

    One-hot encoding explained:
        ML models cannot read text like "ACH" or "Wire".
        We convert one column with 7 possible values
        into 7 binary columns each 0 or 1.

        payment_format = "ACH"  produces:
            fmt_ACH          = 1   <- this one matches
            fmt_Bitcoin      = 0
            fmt_Cash         = 0
            fmt_Cheque       = 0
            fmt_Credit Card  = 0
            fmt_Reinvestment = 0
            fmt_Wire         = 0

    Dictionary comprehension:
        {f"fmt_{fmt}": int(payment_format == fmt) for fmt in ALL_PAYMENT_FORMATS}

        This is shorthand for a for loop building a dict.
        f"fmt_{fmt}" = f-string = inserts variable into the string.
        int(payment_format == fmt) = 1 if match, 0 if not.
    """

    encoded = {}

    # Payment format one-hot: fmt_ACH, fmt_Bitcoin, fmt_Cash, etc.
    for fmt in ALL_PAYMENT_FORMATS:
        encoded[f"fmt_{fmt}"] = int(payment_format == fmt)

    # Receiving currency one-hot — default US Dollar
    for ccy in ALL_CURRENCIES:
        encoded[f"recv_ccy_{ccy}"] = int(ccy == "US Dollar")

    # Payment currency one-hot — default US Dollar
    for ccy in ALL_CURRENCIES:
        encoded[f"pay_ccy_{ccy}"] = int(ccy == "US Dollar")

    return encoded


# ── Main public function ───────────────────────────────────────────

def build_feature_vector(transaction: dict) -> pd.DataFrame:
    """
    Convert a raw transaction dict into a model-ready DataFrame.

    This is the only function api.py needs to call.
    It runs all 4 feature groups and returns 1 row with 66 columns.

    Why return a DataFrame and not a list?
        XGBoost was trained on a DataFrame with named columns.
        Named columns guarantee the model gets the right values
        in the right positions.
        A plain list would risk wrong column order = wrong predictions.

    Args:
        transaction: dict with keys matching the API request schema

    Returns:
        pd.DataFrame with 1 row and 66 columns in training order
    """

    # Extract fields — float() and int() ensure correct types
    amount_paid       = float(transaction["amount_paid"])
    amount_received   = float(transaction["amount_received"])
    hour              = int(transaction["hour"])
    day_of_week       = int(transaction["day_of_week"])
    payment_format    = str(transaction["payment_format"])
    is_same_bank      = int(transaction["is_same_bank"])
    is_cross_currency = int(transaction["is_cross_currency"])
    sender_tx_count   = int(transaction["sender_tx_count"])
    sender_avg_amount = float(transaction["sender_avg_amount"])

    # Build all feature groups and merge into one dict
    # .update() adds all keys from the given dict into features
    features = {}
    features.update(_build_temporal_features(hour, day_of_week))
    features.update(_build_amount_features(amount_paid, amount_received))
    features.update(_build_network_features(is_same_bank, is_cross_currency))
    features.update(_build_velocity_features(
        amount_paid, sender_tx_count, sender_avg_amount
    ))
    features.update(_build_encoded_features(payment_format))

    # pd.DataFrame([features]) = wrap the dict in a list = one row table
    df = pd.DataFrame([features])

    # Reorder columns to match training order exactly
    # reindex fills any missing columns with 0
    try:
        config_path = Path("data/feature_config.json")
        with open(config_path) as f:
            config = json.load(f)
        expected_cols = config["all_features"]
        df = df.reindex(columns=expected_cols, fill_value=0)
    except FileNotFoundError:
        pass

    return df


def validate_payment_format(payment_format: str) -> bool:
    """
    Return True if payment_format is one of the 7 known values.
    Used by api.py to reject bad input before scoring.
    """
    return payment_format in ALL_PAYMENT_FORMATS
