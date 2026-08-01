"""
deploy_endpoint.py
===================
Deploys the Approved XGBoost model package (from Step 4) to a
real-time SageMaker Endpoint for live predictions.
"""
import boto3
import sagemaker
from sagemaker import ModelPackage

REGION = "eu-west-2"
ROLE_ARN = "arn:aws:iam::481088927723:role/AmlShieldSageMakerExecutionRole"
MODEL_PACKAGE_ARN = "arn:aws:sagemaker:eu-west-2:481088927723:model-package/aml-shield-models/1"
ENDPOINT_NAME = "aml-shield-endpoint"

session = sagemaker.Session(boto3.Session(region_name=REGION))

model = ModelPackage(
    role=ROLE_ARN,
    model_package_arn=MODEL_PACKAGE_ARN,
    sagemaker_session=session,
)

print(f"Deploying endpoint '{ENDPOINT_NAME}'... (takes about 5-8 minutes)")
predictor = model.deploy(
    initial_instance_count=2,
    instance_type="ml.m5.large",
    endpoint_name=ENDPOINT_NAME,
)

print(f"Done. Endpoint '{ENDPOINT_NAME}' is live and ready for predictions.")
