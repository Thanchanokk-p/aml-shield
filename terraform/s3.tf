resource "aws_s3_bucket" "aml_shield_data" {
  bucket = "aml-shield-2026"

  tags = {
    Project     = "aml-shield"
    Environment = "portfolio"
    ManagedBy   = "terraform"
  }
}
