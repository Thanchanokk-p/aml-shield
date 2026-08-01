import mlflow
import requests

API_URL = "http://18.134.160.241:8000/predict"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("aml-shield-inference-tests")

test_cases = [
    {
        "name": "low_risk_config",
        "payload": {
            "amount_paid": 50000.00, "amount_received": 50000.00,
            "payment_format": "ACH", "timestamp": "2026-08-01T14:00:00",
            "hour": 14, "day_of_week": 0,
            "is_same_bank": 0, "is_cross_currency": 0,
            "sender_tx_count": 450, "sender_avg_amount": 500.00,
        },
    },
    {
        "name": "medium_risk_config",
        "payload": {
            "amount_paid": 50000.00, "amount_received": 50000.00,
            "payment_format": "ACH", "timestamp": "2026-08-01T02:00:00",
            "hour": 2, "day_of_week": 5,
            "is_same_bank": 0, "is_cross_currency": 0,
            "sender_tx_count": 1, "sender_avg_amount": 500.00,
        },
    },
    {
        "name": "high_risk_config",
        "payload": {
            "amount_paid": 50000.00, "amount_received": 50000.00,
            "payment_format": "ACH", "timestamp": "2026-08-01T02:00:00",
            "hour": 2, "day_of_week": 5,
            "is_same_bank": 1, "is_cross_currency": 0,
            "sender_tx_count": 1, "sender_avg_amount": 45000.00,
        },
    },
]

for case in test_cases:
    response = requests.post(API_URL, json=case["payload"])
    result = response.json()
    risk_score = result["risk_score"]
    risk_level = result["risk_level"]

    with mlflow.start_run(run_name=case["name"]):
        mlflow.log_params(case["payload"])
        mlflow.log_metric("risk_score", risk_score)
        mlflow.set_tag("risk_level", risk_level)
        mlflow.set_tag("source", "ec2_api_live_test")
        print(f"{case['name']}: risk_score={risk_score}, risk_level={risk_level}")

print("Done. Refresh MLflow UI.")
