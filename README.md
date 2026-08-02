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
│   ├── 01_eda_data_exploration.ipynb       EDA on ~5M IBM transactions — class balance, amounts, formats, temporal patterns
│   ├── 02_feature_engineering.ipynb        Builds the canonical 66-feature set → features_engineered.parquet
│   ├── 03_baseline_model_mlflow.ipynb      XGBoost baseline, MLflow-tracked — split, threshold analysis, SHAP, cost model
│   ├── 04_trial.ipynb                      Scratch/experiment notebook (not part of the pipeline)
│   ├── 05_phase8_generalization.ipynb      Re-tests the headline finding on a second dataset — result reverses (p = 0.0044)
│   ├── mlflow.db                           Local MLflow tracking backend (SQLite)
│   ├── mlruns/                             Notebook-scoped MLflow run artifacts
│   └── mlruns.zip                          Archived copy of the above
│
├── src/                                Production Python package (the engineering half)
│   ├── __init__.py
│   ├── api.py                              FastAPI serving app — scores a transaction, runs SHAP, returns JSON
│   ├── features.py                         Single source of truth for feature construction (shared by training + serving)
│   ├── pipeline.py                         Prefect training flow — load → split → train-and-log-model
│   ├── monitoring.py                       Prefect drift flow — Evidently DataDriftPreset, reference vs. current
│   ├── fairness_audit.py                   Per-group AUC across payment formats
│   ├── figures/                            13 exported analysis figures reused in the write-up
│   └── monitoring_reports/
│       └── drift_report.html               Latest generated Evidently drift report
│
├── aws/                                SageMaker migration — the same pipeline, on AWS
│   ├── processing/                         Feature engineering as a SageMaker Processing Job
│   ├── training/                           SageMaker Training Job — built-in XGBoost, same hyperparameters
│   ├── registry/                           Registers + approves the model in the Model Package Group
│   └── deployment/                         Deploys the endpoint; includes the parity-test script (EC2 vs. SageMaker)
│
├── dashboard/                          Streamlit demo front-end
│   ├── app.py                              Transaction form, risk gauge, SHAP reasons — calls the FastAPI backend
│   └── requirements.txt                    streamlit, requests, plotly
│
├── terraform/                          Infrastructure as Code (eu-west-2) — imported, not recreated
│   ├── provider.tf
│   ├── s3.tf
│   ├── iam.tf
│   ├── ec2.tf
│   ├── sagemaker.tf
│   └── .terraform.lock.hcl
│
├── tests/                              The CI gate — deployment fails if these fail
│   ├── test_api_health.py                  API imports cleanly; core routes exist
│   └── test_features.py                    Feature contract — 66 columns, encoding correctness
│
├── models/
│   └── xgboost-model                       Trained XGBoost artifact
│
├── mlruns/                             Project-level MLflow store
│
├── .github/workflows/deploy.yml        CI/CD — test job gates deploy job, auto-rollback on failed health check
├── .devcontainer/devcontainer.json     Codespaces config — auto-runs the dashboard
│
├── Dockerfile                          python:3.12-slim, builds and serves the FastAPI app
├── requirements.txt                    API/runtime dependency pins
├── prefect.yaml                        Scheduled deployments — training + drift-check flows
├── log_inference_tests.py              Fires test payloads at the live endpoint, logs to MLflow
├── DECISION_LOG.md                     Key architectural decisions and why
├── FAILURES.md                         Bugs hit and how they were root-caused
├── .dockerignore / .prefectignore / .gitignore
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
