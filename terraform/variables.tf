# vault provider variables
variable "VAULT_ADDR" {}
variable "APP_ROLE_ID" {}
variable "APP_ROLE_SECRET_ID" {}

# vault data - credentials variables
variable "ACCOUNT_TYPE" {}
variable "VAULT_NAMESPACE_XYZ" {}

# aws provider variales
variable "ATLAS_WORKSPACE_NAME" {}

# MOM default VPC subnet id's
variable "mom_prod_subnet_ids" {
  type = list(string)
}

# MOM default VPC security group id's
variable "mom_prod_security_group_ids" {
  type = list(string)
}

# MOM S3 Resource variables
variable "mom_prod_bucket_name" {
  type = string
}
variable "mom_prod_s3_resource" {
  type = any
}

# MOM IAM resource variables
variable "mom_prod_iam_resource" {
  type = any
}

# MOM Lambda resource variables
variable "mom_prod_lambda_resource" {
  type = any
}
variable "lambda_invoke_action" {
  type = string
}

# MOM KMS resource variables
variable "mom_prod_kms_resource" {
  type = any
}

# MOM API Gateway resource variables
variable "mom_prod_api_gateway_resource" {
  type = any
}

# MOM Event Bridge resource variables
variable "mom_prod_event_bridge_resource" {
  type = any
}

# MOM Cloudwatch resource variables
variable "mom_prod_cloudwatch_resource" {
  type = any
}

