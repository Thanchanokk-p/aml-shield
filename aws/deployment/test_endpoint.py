"""
test_endpoint.py
=================
Sends one sample transaction to the live endpoint and prints
the raw prediction (fraud probability score from XGBoost).

Feature order below matches feature_config.json exactly (66 features,
same order as "all_features" list).
"""
import boto3

REGION = "eu-west-2"
ENDPOINT_NAME = "aml-shield-endpoint"

# Sample: ACH transfer, US Dollar -> US Dollar, $50,000, 2am Saturday,
# different banks, established sender account
sample_values = [
    2,        # hour (2am)
    5,        # day_of_week (Saturday)
    1,        # is_weekend
    1,        # is_night
    0,        # is_business_hrs
    50000,    # Amount Paid
    50000,    # Amount Received
    10.8198,  # log_amount_paid
    10.8198,  # log_amount_received
    0,        # amount_difference
    1.0,      # amount_ratio
    1,        # is_round_100
    1,        # is_round_1000
    1,        # is_round_10000
    3,        # amount_band
    0,        # is_same_bank
    0,        # is_cross_currency
    0,        # is_self_loop
    450,      # sender_tx_count
    500,      # sender_avg_amount
    100,      # sender_std_amount
    1000,     # sender_max_amount
    2,        # sender_unique_banks
    300,      # receiver_tx_count
    600,      # receiver_avg_amount
    100.0,    # amount_vs_sender_avg
    495.0,    # sender_amount_zscore
    1,        # fmt_ACH
    0, 0, 0, 0, 0, 0,  # fmt_Bitcoin, Cash, Cheque, Credit Card, Reinvestment, Wire
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  # recv_ccy_* (US Dollar = 1)
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  # pay_ccy_* (US Dollar = 1)
    0.02,     # from_bank_freq
    0.015,    # to_bank_freq
]

assert len(sample_values) == 66, f"Expected 66 features, got {len(sample_values)}"

sample_csv_row = ",".join(str(v) for v in sample_values)

runtime = boto3.client("sagemaker-runtime", region_name=REGION)

response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="text/csv",
    Body=sample_csv_row,
)

result = response["Body"].read().decode("utf-8")
print(f"Prediction (fraud probability): {result}")
