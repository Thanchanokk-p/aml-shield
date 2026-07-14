"""
src/api.py
===========
The production REST API for AML-Shield.

What this file does:
    Receives HTTP POST requests with transaction data.
    Calls features.py to build the feature vector.
    Runs the XGBoost model to get a fraud probability score.
    Calls SHAP to explain why the model made that decision.
    Returns a structured JSON response.

How to run:
    uvicorn src.api:app --reload --port 8000

How to test:
    Open browser at http://localhost:8000/docs
    FastAPI auto-generates an interactive documentation page.
"""

# ── Imports ────────────────────────────────────────────────────────
import os
import sys
from datetime import datetime

import mlflow.xgboost          # loads our model from MLflow registry
import numpy as np             # array operations for SHAP
import pandas as pd            # DataFrame for feature vector
import shap                    # explains model predictions
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
print(sys.executable)

# Add project root to Python path so we can import src/features.py
# __file__                       = full path to api.py
# dirname(__file__)               = src/ folder
# dirname(dirname(__file__))      = project root (aml-shield-main/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import build_feature_vector, validate_payment_format


# ── App creation ───────────────────────────────────────────────────
app = FastAPI(
    title       = "AML-Shield",
    description = (
        "Real-time Anti-Money Laundering transaction scoring API. "
        "Trained on IBM Research NeurIPS 2023 benchmark (5M transactions). "
        "Returns fraud risk score + SHAP explanation on every request."
    ),
    version  = "1.0.0",
    docs_url = "/docs",
    redoc_url= "/redoc",
)


# ── Plain English labels for SHAP features ────────────────────────
FEATURE_LABELS = {
    "fmt_ACH"              : "ACH payment format (highest risk in this dataset)",
    "sender_unique_banks"  : "Account sends to many different banks",
    "sender_tx_count"      : "Account has high transaction volume",
    "is_night"             : "Transaction at night (10pm to 6am)",
    "is_weekend"           : "Transaction on weekend",
    "is_cross_currency"    : "Different currencies for payment and receiving",
    "amount_vs_sender_avg" : "Amount significantly above account's normal pattern",
    "is_round_1000"        : "Round 1000 amount — possible structuring signal",
    "fmt_Wire"             : "Wire transfer format",
    "fmt_Bitcoin"          : "Bitcoin payment format",
    "fmt_Cash"             : "Cash payment format",
    "is_self_loop"         : "Transaction within same account (safe signal)",
    "fmt_Credit Card"      : "Credit card format (generally safe signal)",
    "Amount Paid"          : "Transaction amount is unusually large",
}


# ── Model loading ──────────────────────────────────────────────────
# Build absolute path to mlflow.db so it works regardless of
# which directory uvicorn is launched from.
#
# Example:
#   api.py lives at: /Users/fripuran/aml-shield-main/src/api.py
#   __file__       = /Users/fripuran/aml-shield-main/src/api.py
#   dirname x1     = /Users/fripuran/aml-shield-main/src
#   dirname x2     = /Users/fripuran/aml-shield-main          <- project root
#   join "mlflow.db" = /Users/fripuran/aml-shield-main/mlflow.db
#
# Using "sqlite:///mlflow.db" (relative path) was failing because
# uvicorn resolves relative paths from wherever it was launched,
# not from where api.py lives.

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH      = os.path.join(_PROJECT_ROOT, "mlflow.db")
_TRACKING_URI = f"sqlite:///{_DB_PATH}"

print("Loading model from MLflow registry...")
print(f"  Project root : {_PROJECT_ROOT}")
print(f"  MLflow DB    : {_DB_PATH}")
print(f"  DB exists    : {os.path.exists(_DB_PATH)}")

try:
    mlflow.set_tracking_uri(_TRACKING_URI)
    model     = mlflow.xgboost.load_model("models:/aml-shield-xgboost/1")
    explainer = shap.TreeExplainer(model)
    MODEL_LOADED = True
    print("Model loaded successfully.")

except Exception as e:
    print(f"WARNING: Model failed to load: {e}")
    model        = None
    explainer    = None
    MODEL_LOADED = False


# ── Request and Response Schemas ───────────────────────────────────

class Transaction(BaseModel):
    """Transaction data sent by the client for fraud scoring."""

    amount_paid       : float = Field(..., gt=0,  description="Amount sent (currency units)")
    amount_received   : float = Field(..., gt=0,  description="Amount received")
    hour              : int   = Field(..., ge=0,  le=23, description="Hour of day 0-23")
    day_of_week       : int   = Field(..., ge=0,  le=6,  description="0=Monday 6=Sunday")
    payment_format    : str   = Field(..., description="ACH | Wire | Cash | Cheque | Credit Card | Bitcoin | Reinvestment")
    is_same_bank      : int   = Field(..., ge=0,  le=1,  description="1 if same bank for both accounts")
    is_cross_currency : int   = Field(..., ge=0,  le=1,  description="1 if currencies differ")
    sender_tx_count   : int   = Field(..., gt=0,  description="Total transactions sent by this account")
    sender_avg_amount : float = Field(..., gt=0,  description="Average amount this account normally sends")

    model_config = {
        "json_schema_extra": {
            "example": {
                "amount_paid"       : 98000,
                "amount_received"   : 97500,
                "hour"              : 2,
                "day_of_week"       : 6,
                "payment_format"    : "ACH",
                "is_same_bank"      : 0,
                "is_cross_currency" : 0,
                "sender_tx_count"   : 450,
                "sender_avg_amount" : 500,
            }
        }
    }


class PredictionResponse(BaseModel):
    """JSON response returned by /predict."""
    transaction_id : str
    risk_score     : float
    flagged        : bool
    risk_level     : str
    top_reasons    : list
    threshold_used : float
    timestamp      : str


# ── In-memory counters ─────────────────────────────────────────────
_stats = {
    "total"   : 0,
    "flagged" : 0,
    "start"   : datetime.now(),
}


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/")
def root():
    """Root endpoint — confirms the API is running."""
    return {
        "service"      : "AML-Shield",
        "version"      : "1.0.0",
        "status"       : "running",
        "model_loaded" : MODEL_LOADED,
        "dataset"      : "IBM NeurIPS 2023 AML Benchmark",
        "training_auc" : 0.9845,
        "docs"         : "/docs",
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint used by cloud platforms.
    DigitalOcean and AWS call this every 30 seconds.
    Non-200 response triggers automatic container restart.
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code = 503,
            detail      = "Model not loaded. Service degraded."
        )
    return {
        "status"       : "healthy",
        "model_loaded" : MODEL_LOADED,
        "timestamp"    : datetime.now().isoformat(),
    }


@app.get("/metrics")
def get_metrics():
    """Basic inference statistics for monitoring."""
    total     = _stats["total"]
    flagged   = _stats["flagged"]
    flag_rate = (flagged / total * 100) if total > 0 else 0.0
    uptime    = (datetime.now() - _stats["start"]).total_seconds()

    return {
        "total_predictions" : total,
        "total_flagged"     : flagged,
        "flag_rate_pct"     : round(flag_rate, 4),
        "uptime_seconds"    : round(uptime, 1),
        "model_version"     : "aml-shield-xgboost/1",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction, threshold: float = 0.5):
    """
    Score a transaction for money laundering risk.

    threshold (default 0.5):
        Lower = catch more fraud = more false alarms.
        Higher = fewer false alarms = may miss fraud.
        Cost model shows 0.75 is optimal for this dataset,
        saving £1.25M vs the default 0.5.
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code = 503,
            detail      = "Model not available. Check /health."
        )

    if not validate_payment_format(transaction.payment_format):
        raise HTTPException(
            status_code = 400,
            detail      = (
                f"Unknown payment format: '{transaction.payment_format}'. "
                "Valid: ACH, Wire, Cash, Cheque, Credit Card, Bitcoin, Reinvestment"
            )
        )

    try:
        # Step 1: Build 66-feature vector
        features = build_feature_vector(transaction.model_dump())

        # Step 2: Get fraud probability (0.0 to 1.0)
        risk_score = float(model.predict_proba(features)[0][1])

        # Step 3: Apply threshold
        flagged = risk_score >= threshold

        # Step 4: Assign risk level
        if   risk_score >= 0.80: risk_level = "CRITICAL"
        elif risk_score >= 0.50: risk_level = "HIGH"
        elif risk_score >= 0.30: risk_level = "MEDIUM"
        else:                    risk_level = "LOW"

        # Step 5: SHAP explanation
        shap_values = explainer.shap_values(features)
        top_reasons = _get_top_reasons(features, shap_values, n=3)

        # Step 6: Update counters
        _stats["total"] += 1
        if flagged:
            _stats["flagged"] += 1

        # Step 7: Return response
        txn_id = f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:18]}"

        return PredictionResponse(
            transaction_id = txn_id,
            risk_score     = round(risk_score, 4),
            flagged        = flagged,
            risk_level     = risk_level,
            top_reasons    = top_reasons,
            threshold_used = threshold,
            timestamp      = datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Private helper ─────────────────────────────────────────────────

def _get_top_reasons(
    features   : pd.DataFrame,
    shap_values: list,
    n          : int = 3,
) -> list:
    """
    Convert raw SHAP values into top N plain-English reasons.

    SHAP value meaning:
        Positive = feature pushed score toward FRAUD
        Negative = feature pushed score toward LEGIT
        We sort by absolute value to find most important features.
    """
    feature_names = features.columns.tolist()
    shap_abs      = np.abs(shap_values[0])
    top_indices   = np.argsort(shap_abs)[-n:][::-1]

    reasons = []
    for idx in top_indices:
        fname     = feature_names[idx]
        fval      = float(features.iloc[0][fname])
        shap_val  = float(shap_values[0][idx])
        direction = "increases" if shap_val > 0 else "decreases"
        label     = FEATURE_LABELS.get(fname, fname.replace("_", " ").title())

        reasons.append({
            "feature"    : fname,
            "label"      : label,
            "value"      : round(fval, 4),
            "direction"  : direction,
            "shap_value" : round(shap_val, 4),
        })

    return reasons