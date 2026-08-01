"""
launch_job.py
=============
Launches the feature engineering script above as a SageMaker
Processing Job. Run this from your Mac — it only submits the job
and waits; the actual computation happens on AWS.
"""
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

ROLE_ARN = "arn:aws:iam::481088927723:role/AmlShieldSageMakerExecutionRole"

BUCKET = "aml-shield-2026"

processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=ROLE_ARN,
    instance_type="ml.m5.2xlarge",
    instance_count=1,
    base_job_name="aml-shield-feature-eng",
)

print("Submitting SageMaker Processing Job...")
processor.run(
    code="feature_engineering_job.py",
    inputs=[
        ProcessingInput(
            source=f"s3://{BUCKET}/raw/HI-Small_Trans.csv",
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{BUCKET}/processed/",
        )
    ],
)
print("Job complete.")
