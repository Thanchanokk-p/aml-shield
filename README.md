# AML-Shield
**An end-to-end money-laundering detection system — from a Jupyter notebook to a live AWS MLOps pipeline.**

[![CI](https://github.com/Thanchanokk-p/aml-shield/actions/workflows/deploy.yml/badge.svg)](https://github.com/Thanchanokk-p/aml-shield/actions)
[![Live Demo](https://img.shields.io/badge/demo-streamlit-FF4B4B)](https://aml-shield-mlops.streamlit.app/)

AML-Shield is a full-cycle MLOps project built on IBM's NeurIPS 2023 Anti-Money Laundering benchmark (~5M transactions, 0.10% fraud rate). It covers everything from exploratory data analysis to a production deployment on AWS SageMaker, including a real production bug (a cross-environment parity failure) that shaped most of the engineering decisions in the second half of the project.

**Live demo:** [https://aml-shield-mlops.streamlit.app/](https://aml-shield-mlops.streamlit.app/)

---

## What This Project Actually Demonstrates

- **Modeling under severe class imbalance** (980:1) — XGBoost, AUC-ROC 0.9831, with an honest breakdown of why AUC-ROC alone is misleading at this imbalance level (Precision sits at 1.8% at the default threshold)
- **A generalization study** that stress-tested a promising finding against a second, independent dataset, and watched it reverse (p = 0.0044)
- **A real production incident**: two "identical" deployments (EC2 and SageMaker) returned different risk scores for the same transaction (0.6807 vs 0.5912). Root-caused to a stale model reference (training-serving skew), fixed, and proven fixed with an automated parity test, not just a one-time manual check
- **Infrastructure as Code**: existing hand-built AWS resources imported into Terraform with zero destructive changes
- **CI/CD with a real test gate**: deployment is blocked unless the test suite passes
- **Explainability and fairness**: SHAP-based plain-English explanations, capped to reflect how many reasons a human reviewer can actually use; a fairness audit across payment formats

---

## Repository Structure

```text
aml-shield/
│
├── notebooks/                          Exploratory + experimental work (the research half)
│   ├── 01_eda_data_exploration.ipynb       EDA on ~5M IBM transactions: class balance, amount
│   │                                        distributions, payment formats, currencies, temporal
│   │                                        and velocity patterns (produces figures 01–07)
│   ├── 02_feature_engineering.ipynb        Builds the canonical 66-feature set: temporal, amount,
│   │                                        network, velocity and one-hot encoded features.
│   │                                        Writes features_engineered.parquet + feature_config.json
│   ├── 03_baseline_model_mlflow.ipynb      XGBoost baseline, MLflow-tracked. Train/val/test split
│   │                                        (70/15/15, random_state=42), threshold analysis,
│   │                                        SHAP summary + waterfall, business cost model
│   ├── 04_trial.ipynb                      Scratch/experiment notebook (not part of the pipeline)
│   ├── 05_phase8_generalization.ipynb      Phase 8: re-tests the headline finding on a second,
│   │                                        independent dataset — the result reverses (p = 0.0044)
│   ├── mlflow.db                           Local MLflow tracking backend (SQLite) used by notebooks
│   ├── mlruns/                             Notebook-scoped MLflow run artifacts (figures, models)
│   └── mlruns.zip                          Archived copy of the above
│
├── src/                                Production Python package (the engineering half)
│   ├── __init__.py
│   ├── api.py                              FastAPI serving app. Loads the XGBoost model from the
│   │                                        MLflow registry via an absolute sqlite path, scores a
│   │                                        transaction, runs SHAP, and returns JSON.
│   │                                        Routes: GET / , GET /health , GET /metrics ,
│   │                                        POST /predict?threshold=0.5
│   │                                        Helpers: _get_top_reasons, _format_reason_sentence,
│   │                                        build_explanation_summary (plain-English SHAP output)
│   ├── features.py                         The single source of truth for feature construction —
│   │                                        the notebook logic rewritten as importable functions so
│   │                                        training and serving cannot drift apart.
│   │                                        _build_temporal_features, _build_amount_features,
│   │                                        _build_network_features, _build_velocity_features,
│   │                                        _build_encoded_features → build_feature_vector (66 cols),
│   │                                        plus validate_payment_format.
│   │                                        Pins ALL_PAYMENT_FORMATS (7) and ALL_CURRENCIES (15) so
│   │                                        one-hot columns always match training order
│   ├── pipeline.py                         Prefect training flow (`aml-shield-training-pipeline`) —
│   │                                        notebook 03 sections 0–3 converted to tasks:
│   │                                        load-data (2 retries) → split-data → train-and-log-model
│   │                                        (1 retry), logging params/metrics to MLflow
│   ├── monitoring.py                       Prefect drift flow (`aml-shield-drift-check`) — Evidently
│   │                                        DataDriftPreset comparing an older 70% reference slice
│   │                                        against a newer 30% current slice
│   ├── fairness_audit.py                   Per-group AUC across one-hot payment-format columns
│   │                                        (fmt_*), using the latest MLflow run's model
│   ├── figures/                            13 exported analysis figures (01_class_distribution.png …
│   │                                        13_shap_waterfall_fraud.png) reused in the write-up
│   └── monitoring_reports/
│       └── drift_report.html               Latest generated Evidently drift report
│
├── aws/                                SageMaker migration — the same pipeline, on AWS (Phase 9)
│   ├── processing/
│   │   ├── feature_engineering_job.py      Runs as a SageMaker Processing Job: reads raw
│   │   │                                    HI-Small_Trans.csv from /opt/ml/processing/input/,
│   │   │                                    applies the identical 5 feature functions, writes
│   │   │                                    parquet + feature_config.json to the output mount
│   │   └── launch_job.py                   Submits that job (SKLearnProcessor, ml.m5.2xlarge)
│   ├── training/
│   │   ├── prepare_training_data.py        Converts the parquet to SageMaker XGBoost CSV format
│   │   │                                    (target first, no header/index); same 70/15 split and
│   │   │                                    random_state=42 as the local run; emits scale_pos_weight
│   │   ├── launch_training.py              SageMaker Training Job with built-in XGBoost 1.7-1 and
│   │   │                                    hyperparameters matched to notebook 03
│   │   └── scale_pos_weight.txt            979.9163907284768 — the measured imbalance ratio
│   ├── registry/
│   │   └── register_model.py               Registers model.tar.gz into the SageMaker Model Package
│   │                                        Group "aml-shield-models" and approves it
│   └── deployment/
│       ├── deploy_endpoint.py              Deploys the approved model package to the real-time
│       │                                    endpoint `aml-shield-endpoint` (eu-west-2)
│       ├── test_endpoint.py                Sends one hand-built 66-value CSV row and prints the
│       │                                    raw probability
│       ├── test_endpoint_showcase.py       Side-by-side scoring of several sample transactions
│       └── build_high_risk_sample.py       The parity test: builds the vector with the REAL
│                                            src.features.build_feature_vector and sends it to
│                                            SageMaker, so EC2 and SageMaker can be compared on
│                                            genuinely identical input
│
├── dashboard/                          Streamlit demo front-end
│   ├── app.py                              Transaction entry form, health banner, Plotly risk gauge,
│   │                                        SHAP reasons — calls the EC2 FastAPI backend
│   └── requirements.txt                    streamlit, requests, plotly (deployed separately from the API)
│
├── terraform/                          Infrastructure as Code (eu-west-2)
│   ├── provider.tf                         Terraform + AWS provider config
│   ├── s3.tf                               aws_s3_bucket.aml_shield_data (bucket: aml-shield-2026)
│   ├── iam.tf                              SageMaker execution role + SageMakerFullAccess and
│   │                                        S3FullAccess policy attachments
│   ├── ec2.tf                              aws_instance.aml_shield_server — the API host
│   ├── sagemaker.tf                        aws_sagemaker_model_package_group.aml_shield_models
│   └── .terraform.lock.hcl                 Provider version lock
│                                           (state files are gitignored — imported, never recreated)
│
├── tests/                              The CI gate — deployment fails if these fail
│   ├── __init__.py
│   ├── test_api_health.py                  API imports cleanly; /predict and /health routes exist
│   └── test_features.py                    Feature contract: 66 columns out; is_weekend for
│                                            Sat/Mon; is_night at 2am; ACH one-hot encoding
│
├── models/
│   └── xgboost-model                       Trained XGBoost artifact (~1.1 MB)
│
├── mlruns/                             Project-level MLflow store (local file backend)
│   ├── 0/ , 1/                             Experiment runs with figure artifacts and model versions
│   ├── 852688675953394998/                 Run with the full metric set: test/val AUC-ROC,
│   │                                        avg_precision, F1, precision, recall, TP/FP/TN/FN,
│   │                                        plus all XGBoost hyperparameters
│   └── models/aml-shield-xgboost/          Registered model, version-1
│
├── .github/
│   └── workflows/
│       └── deploy.yml                      CI/CD on push to main:
│                                            job 1 `test` — pytest tests/ -v
│                                            job 2 `deploy` (needs: test) — tag current image as
│                                            rollback point, SSH to EC2, pull, disk-space guard
│                                            (prune below 2GB), docker build/run, retrying health
│                                            check (6 × 5s), auto-rollback to :previous on failure
│
├── .devcontainer/
│   └── devcontainer.json                   Codespaces config — installs requirements and auto-runs
│                                            the Streamlit dashboard on port 8501
│
├── Dockerfile                          python:3.12-slim, g++ for SHAP's C++ extensions,
│                                        requirements-first layer caching, copies src/ + mlflow.db +
│                                        mlruns/, exposes 8000, runs uvicorn src.api:app
├── requirements.txt                    API/runtime pins: fastapi 0.115.6, uvicorn 0.34.0,
│                                        pydantic 2.10.4, xgboost 1.7.6, scikit-learn 1.3.2,
│                                        mlflow 2.9.2, shap 0.44.1, pandas 2.1.4, numpy 1.26.4
├── prefect.yaml                        Two scheduled deployments on `local-pool`:
│                                        aml-shield-training (cron 0 19 * * *, Europe/London) and
│                                        aml-shield-drift-check (cron 30 20 * * *, Europe/London)
├── log_inference_tests.py              Fires low/medium/high-risk payloads at the live EC2 /predict
│                                        endpoint and logs each score to the MLflow experiment
│                                        `aml-shield-inference-tests` — this is what turns the parity
│                                        check into a repeatable test rather than a screenshot
├── DECISION_LOG.md                     Key architectural decisions and why
├── FAILURES.md                         Bugs hit and how they were root-caused
├── .dockerignore                       Keeps notebooks/, data/, tests/ and *.md out of the image
├── .prefectignore                      Excludes caches, envs and editor files from flow uploads
├── .gitignore                          Excludes data/, *.csv, mlflow.db, terraform state, venvs
├── src.zip                             Archived snapshot of src/
└── README.md
```
---

## Key Results

| Metric | Value |
|---|---|
| AUC-ROC (test set) | 0.9831 |
| Recall (fraud caught) | 88.2% |
| Precision at threshold 0.5 | 1.8% (54 false alarms per fraud caught) |
| Cross-dataset generalization (AUC-ROC) | 0.9831 → 0.9687 |
| Fairness audit (AUC spread across payment formats) | 0.0356 |
| Data drift monitored | 8/66 columns (12.1%), below the 50% alert threshold |
| Parity test (EC2 vs SageMaker, post-fix) | 0.5912 = 0.5912, exact match |

---

## Running It Locally

```bash
# clone and set up
git clone https://github.com/Thanchanokk-p/aml-shield.git
cd aml-shield
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run the API
python src/api.py

# run the dashboard (in a second terminal)
streamlit run dashboard/app.py
```

### Automated Pipelines (Optional)

```bash
# Prefect (training + drift-check flows)
prefect server start

# in a second terminal
prefect work-pool create local-pool --type process
prefect deploy --all
prefect worker start --pool local-pool
```

---

## AWS Deployment

The production stack (SageMaker Processing/Training/Registry/Endpoint, Terraform-managed infrastructure) is documented in `aws/` and `terraform/`. See `DECISION_LOG.md` for the reasoning behind each major infrastructure choice, and `FAILURES.md` for a running log of production bugs encountered and how each was diagnosed, including the cross-environment parity bug that anchors the full write-up.

---

## Dataset

[IBM NeurIPS 2023 Anti-Money Laundering benchmark](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) — synthetic, CDLA-Sharing-1.0 licensed. Not redistributed in this repository; see the notebooks for the exact loading and preprocessing steps.
