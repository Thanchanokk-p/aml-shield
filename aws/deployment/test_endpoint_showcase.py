"""
test_endpoint_showcase.py
===========================
Sends multiple sample transactions to the live SageMaker Endpoint
and prints a side-by-side comparison — for content/demo purposes.
"""
import boto3

REGION = "eu-west-2"
ENDPOINT_NAME = "aml-shield-endpoint"

runtime = boto3.client("sagemaker-runtime", region_name=REGION)


def predict(sample_values, label):
    assert len(sample_values) == 66, f"{label}: expected 66, got {len(sample_values)}"
    csv_row = ",".join(str(v) for v in sample_values)
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=csv_row,
    )
    score = float(response["Body"].read().decode("utf-8").strip())
    print(f"{label:35s} -> risk_score: {score:.4f}")
    return score


# Baseline: normal-looking transaction
baseline = [
    14, 2, 0, 0, 1, 500, 500, 6.2166, 6.2166, 0, 1.0, 1, 0, 0, 1,
    1, 0, 0, 450, 500, 100, 1000, 2, 300, 600, 100.0, 495.0,
    0, 0, 0, 0, 0, 0, 1,
    0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
    0.02, 0.015,
]

# Aggressive: extreme values pushed toward known-important features
aggressive = [
    2, 5, 1, 1, 0, 950000, 950000, 13.7643, 13.7643, 0, 1.0, 0, 0, 0, 4,
    0, 1, 0, 2, 1.0, 0.5, 5, 1, 5, 10.0, 949999.0, 949998.0,
    1, 0, 0, 0, 0, 0, 0,
    0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
    0.001, 0.001,
]

print("=" * 60)
print("AML-Shield — Live SageMaker Endpoint Prediction Test")
print("=" * 60)
predict(baseline, "Baseline (typical transaction)")
predict(aggressive, "Aggressive (ACH, $950k, 2am, new account)")
print("=" * 60)
