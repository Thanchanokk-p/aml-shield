"""
register_model.py
==================
Registers the trained XGBoost model artifact (from Step 3) into a
SageMaker Model Package Group ("Model Registry"), then approves it
so it's ready to deploy in Step 6 (Endpoint).
"""
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker import image_uris

REGION = "eu-west-2"
BUCKET = "aml-shield-2026"
ROLE_ARN = "arn:aws:iam::481088927723:role/AmlShieldSageMakerExecutionRole"

# Exact S3 path to the model.tar.gz produced by Step 3
MODEL_DATA = (
    f"s3://{BUCKET}/models/aml-shield-xgboost-2026-07-29-22-54-02-572/output/model.tar.gz"
)

MODEL_PACKAGE_GROUP_NAME = "aml-shield-models"

session = sagemaker.Session(boto3.Session(region_name=REGION))

xgboost_image_uri = image_uris.retrieve(
    framework="xgboost", region=REGION, version="1.7-1"
)

model = Model(
    image_uri=xgboost_image_uri,
    model_data=MODEL_DATA,
    role=ROLE_ARN,
    sagemaker_session=session,
)

# Create the Model Package Group if it doesn't exist yet
sm_client = boto3.client("sagemaker", region_name=REGION)
existing_groups = [
    g["ModelPackageGroupName"]
    for g in sm_client.list_model_package_groups()["ModelPackageGroupSummaryList"]
]
if MODEL_PACKAGE_GROUP_NAME not in existing_groups:
    sm_client.create_model_package_group(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
        ModelPackageGroupDescription="AML-Shield fraud detection models (XGBoost, LR)",
    )
    print(f"Created Model Package Group: {MODEL_PACKAGE_GROUP_NAME}")
else:
    print(f"Model Package Group already exists: {MODEL_PACKAGE_GROUP_NAME}")

print("Registering XGBoost model...")
model_package = model.register(
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.large", "ml.m5.xlarge"],
    transform_instances=["ml.m5.large"],
    model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
    approval_status="Approved",  # auto-approve XGBoost per plan
)

print(f"Done. Model Package ARN:\n{model_package.model_package_arn}")
