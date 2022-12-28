# local block to read nested values from mom_prod_kms_resource map object
locals {
  s3_service     = var.mom_prod_kms_resource["service_names"]["s3_bucket"]
  lambda_service = var.mom_prod_kms_resource["service_names"]["lambda_function"]
}

# Module for generating KMS key which to be attached to LAMBDA functions and S3 Bucket
module "mom_prod_aws_kms_lambda_s3" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-kms-service/aws"
  version = "0.1.8"

  application      = var.application-name
  application_dl   = var.app-support-dl
  barometer-it-num = var.apm-id
  company          = var.company
  compliance       = var.compliance
  costcenter       = var.costcenter
  create_kms_key   = var.mom_prod_kms_resource["create_kms_key"]
  description      = var.mom_prod_kms_resource["description"]
  environment      = var.environment
  it-department    = var.business-division
  layer            = var.PatchWindow
  owner-department = var.app-servicenow-group
  resource-type    = var.mom_prod_kms_resource["resource_type"]
  service_name     = [local.lambda_service, local.s3_service]
  multi_region     = var.mom_prod_kms_resource["multi_region"]
}