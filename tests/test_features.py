import sys
sys.path.insert(0, "/Users/fripuran/aml-shield-main")
from src.features import build_feature_vector

BASE_TX = {
    "amount_paid": 1000.0, "amount_received": 1000.0,
    "hour": 10, "day_of_week": 0, "payment_format": "ACH",
    "is_same_bank": 0, "is_cross_currency": 0,
    "sender_tx_count": 10, "sender_avg_amount": 500.0,
}

def test_output_has_66_columns():
    df = build_feature_vector(BASE_TX)
    assert df.shape == (1, 66)

def test_is_weekend_true_for_saturday():
    tx = {**BASE_TX, "day_of_week": 5}
    df = build_feature_vector(tx)
    assert df["is_weekend"].iloc[0] == 1

def test_is_weekend_false_for_monday():
    tx = {**BASE_TX, "day_of_week": 0}
    df = build_feature_vector(tx)
    assert df["is_weekend"].iloc[0] == 0

def test_is_night_true_for_2am():
    tx = {**BASE_TX, "hour": 2}
    df = build_feature_vector(tx)
    assert df["is_night"].iloc[0] == 1

def test_payment_format_one_hot_ach():
    tx = {**BASE_TX, "payment_format": "ACH"}
    df = build_feature_vector(tx)
    assert df["fmt_ACH"].iloc[0] == 1
    assert df["fmt_Wire"].iloc[0] == 0
