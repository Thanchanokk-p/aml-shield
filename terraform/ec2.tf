resource "aws_instance" "aml_shield_server" {
  ami           = "ami-07f936ee1f9a0de0e"  # Ubuntu 24.04 LTS, eu-west-2
  instance_type = "t3.micro"
  key_name      = "aml-shield-key"

  tags = {
    Name      = "aml-shield-server"
    Project   = "aml-shield"
    ManagedBy = "terraform"
  }

  lifecycle {
    ignore_changes = [ami]
  }
}
