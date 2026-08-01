resource "aws_sagemaker_model_package_group" "aml_shield_models" {
  model_package_group_name        = "aml-shield-models"
  model_package_group_description = "AML-Shield fraud detection models (XGBoost, LR)"

  tags = {
    Project   = "aml-shield"
    ManagedBy = "terraform"
  }
}
