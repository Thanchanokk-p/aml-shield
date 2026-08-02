# AML-Shield
**An end-to-end money-laundering detection system — from a Jupyter notebook to a live AWS MLOps pipeline.**

[![CI](https://github.com/Thanchanokk-p/aml-shield/actions/workflows/deploy.yml/badge.svg)](https://github.com/Thanchanokk-p/aml-shield/actions)
[![Live Demo](https://img.shields.io/badge/demo-streamlit-FF4B4B)](https://aml-shield-dthrnzymsmqrcehjed22i2.streamlit.app)

AML-Shield is a full-cycle MLOps project built on IBM's NeurIPS 2023 Anti-Money Laundering benchmark (~5M transactions, 0.10% fraud rate). It covers everything from exploratory data analysis to a production deployment on AWS SageMaker, including a real production bug (a cross-environment parity failure) that shaped most of the engineering decisions in the second half of the project.

**Full write-up:** link to TDS article once published
**Live demo:** [aml-shield-dthrnzymsmqrcehjed22i2.streamlit.app](https://aml-shield-dthrnzymsmqrcehjed22i2.streamlit.app)

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
├── notebooks/              EDA, feature engineering, baseline model, generalization study
├── src/                    FastAPI serving code, feature engineering, monitoring, pipeline
├── dashboard/               Streamlit demo app (calls the EC2 API)
├── aws/                    SageMaker deployment/processing/training/registry scripts
├── terraform/               Infrastructure as Code (S3, IAM, EC2, SageMaker Model Registry)
├── tests/                  Unit tests (gate CI deployment)
├── models/                  Trained model artifacts
├── mlruns/                  MLflow experiment tracking (local backend)
├── .github/workflows/        CI/CD pipeline (test then deploy, gated)
├── Dockerfile
├── requirements.txt
├── prefect.yaml              Prefect flow deployment config
├── log_inference_tests.py     Live API parity-test logger
├── DECISION_LOG.md           Key architectural decisions and why
├── FAILURES.md               Bugs hit and how they were root-caused
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
