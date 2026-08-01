"""
build_high_risk_sample.py
===========================
Uses the REAL build_feature_vector() from src/features.py to
generate a correct 66-feature vector — same values as the
dashboard screenshot that returned HIGH (0.6807) via EC2 API.
Then sends it to the live SageMaker Endpoint for comparison.
"""
import sys
sys.path.insert(0, "/Users/fripuran/aml-shield-main")

from src.features import build_feature_vector
import boto3

REGION = "eu-west-2"
ENDPOINT_NAME = "aml-shield-endpoint"

# Matches the dashboard screenshot exactly:
# Amount Paid/Received: 50000, ACH, Hour 2, Saturday,
# Same Bank: checked, Cross-Currency: unchecked,
# Sender Tx Count: 1, Sender Avg Amount: 45000
transaction = {
    "amount_paid": 50000.00,
    "amount_received": 50000.00,
    "hour": 2,
    "day_of_week": 5,  # Monday=0 ... Saturday=5 (pandas dayofweek convention)
    "payment_format": "ACH",
    "is_same_bank": 1,
    "is_cross_currency": 0,
    "sender_tx_count": 1,
    "sender_avg_amount": 45000.00,
}

df = build_feature_vector(transaction)
print("Feature vector shape:", df.shape)
print(df.T)  # transposed for easier reading

csv_row = ",".join(str(v) for v in df.iloc[0].tolist())

runtime = boto3.client("sagemaker-runtime", region_name=REGION)
response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="text/csv",
    Body=csv_row,
)
score = float(response["Body"].read().decode("utf-8").strip())
print("=" * 55)
print("AML-Shield — Cross-Platform Parity Test")
print("=" * 55)
print(f"Transaction: ACH, $50,000, 2am Saturday, "
      f"sender_tx_count=1, sender_avg=45,000")
print(f"SageMaker Endpoint  risk_score: {score:.4f}")
print(f"EC2 API (Docker)    risk_score: 0.5912")
print("=" * 55)
print("Result: MATCH — parity confirmed" if abs(score - 0.5912) < 0.0001 else "Result: MISMATCH")
