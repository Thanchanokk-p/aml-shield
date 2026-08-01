"""
launch_training.py
===================
Launches a SageMaker Training Job using the built-in XGBoost
algorithm. Hyperparameters match 03_baseline_model_mlflow.ipynb
exactly, for a fair comparison against the locally trained model.
"""
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

ROLE_ARN = "arn:aws:iam::481088927723:role/AmlShieldSageMakerExecutionRole"
BUCKET = "aml-shield-2026"
REGION = "eu-west-2"

with open("scale_pos_weight.txt") as f:
    scale_pos_weight = f.read().strip()

xgboost_image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=REGION,
    version="1.7-1",
)

estimator = Estimator(
    image_uri=xgboost_image_uri,
    role=ROLE_ARN,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"s3://{BUCKET}/models/",
    base_job_name="aml-shield-xgboost",
)

estimator.set_hyperparameters(
    objective="binary:logistic",
    num_round="300",
    max_depth="6",
    min_child_weight="5",
    subsample="0.8",
    colsample_bytree="0.8",
    eta="0.05",
    gamma="1",
    scale_pos_weight=scale_pos_weight,
    seed="42",
    eval_metric="aucpr",
    tree_method="hist",
)

train_input = TrainingInput(
    s3_data=f"s3://{BUCKET}/training-data/train.csv", content_type="text/csv"
)
val_input = TrainingInput(
    s3_data=f"s3://{BUCKET}/training-data/validation.csv", content_type="text/csv"
)

print("Submitting SageMaker Training Job...")
estimator.fit({"train": train_input, "validation": val_input})
print("Training complete.")
