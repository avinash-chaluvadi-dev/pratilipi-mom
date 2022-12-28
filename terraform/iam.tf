# local block to read iam roles/values from mom_prod_iam_resource map object
locals {
  mom_prod_iam_resource               = var.mom_prod_iam_resource
  role_policies                       = local.mom_prod_iam_resource["role_policies"]
  mom_kms_access_policy               = local.mom_prod_iam_resource["mom_kms_access_policy"]
  mom_api_gw_assume_role              = local.mom_prod_iam_resource["mom_api_gw_assume_role"]
  mom_lambda_assume_role              = local.mom_prod_iam_resource["mom_lambda_assume_role"]
  mom_transcribe_access_policy        = local.mom_prod_iam_resource["mom_transcribe_access_policy"]
  mom_ec2_modify_attribute_policy     = local.mom_prod_iam_resource["mom_ec2_modify_attribute_policy"]
  mom_s3_read_write_access_policy     = local.mom_prod_iam_resource["mom_s3_read_write_access_policy"]
  mom_api_gw_cloudwatch_access_policy = local.mom_prod_iam_resource["mom_api_gw_cloudwatch_access_policy"]

  // Local variables correspond to IAM custom policies
  mom_kms_access_policy_file               = local.mom_kms_access_policy["file_name"]
  mom_transcribe_access_policy_file        = local.mom_transcribe_access_policy["file_name"]
  mom_ec2_modify_attribute_policy_file     = local.mom_ec2_modify_attribute_policy["file_name"]
  mom_s3_read_write_access_policy_file     = local.mom_s3_read_write_access_policy["file_name"]
  mom_api_gw_cloudwatch_access_policy_file = local.mom_api_gw_cloudwatch_access_policy["file_name"]

}

# Data block to read EC2 ModifyInstanceAttribute policy from local filesystem
data "local_file" "mom_prod_ec2_modify_attribute_policy" {
  filename = "${local.role_policies}/${local.mom_ec2_modify_attribute_policy_file}"
}
# Data block to read KMS access policy from local filesystem
data "local_file" "mom_prod_kms_access_policy" {
  filename = "${local.role_policies}/${local.mom_kms_access_policy_file}"
}

# Data block to read S3 read/write access policy from local filesystem
data "local_file" "mom_prod_s3_read_write_access_policy" {
  filename = "${local.role_policies}/${local.mom_s3_read_write_access_policy_file}"
}

# Data block to read transcribe access policy from local filesystem
data "local_file" "mom_prod_transcribe_access_policy" {
  filename = "${local.role_policies}/${local.mom_transcribe_access_policy_file}"
}

# Data block to read mom_prod_api_gw_cloudwatch access policy from local filesystem
data "local_file" "mom_prod_api_gw_cloudwatch_access_policy" {
  filename = "${local.role_policies}/${local.mom_api_gw_cloudwatch_access_policy_file}"
}

data "template_file" "mom_prod_kms_access_policy_template" {
  template = data.local_file.mom_prod_kms_access_policy.content
  vars = {
    s3_kms_key_arn     = lookup(module.mom_prod_aws_kms_lambda_s3.kms_arn, local.s3_service)
    lambda_kms_key_arn = lookup(module.mom_prod_aws_kms_lambda_s3.kms_arn, local.lambda_service)
  }
}

// Module for EC2 ModifyInstanceAttribute policy to be attached for following IAM Role --> mom_prod_aws_lambda_assume_role
module "mom_prod_aws_ec2_modify_attribute_policy" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-iam-policy/aws"
  version = "0.0.9"

  application           = var.application-name
  application_dl        = var.app-support-dl
  aws_create_iam_policy = local.mom_ec2_modify_attribute_policy["create_policy"]
  barometer-it-num      = var.apm-id
  company               = var.company
  compliance            = var.compliance
  costcenter            = var.costcenter
  description           = local.mom_ec2_modify_attribute_policy["description"]
  environment           = var.environment
  it-department         = var.business-division
  layer                 = var.PatchWindow
  name                  = local.mom_ec2_modify_attribute_policy["policy_name"]
  owner-department      = var.app-servicenow-group
  policy                = data.local_file.mom_prod_ec2_modify_attribute_policy.content
  resource-type         = local.mom_prod_iam_resource["resource_type"]
}

// Module for KMS access policy to be attached for following IAM Role --> mom_prod_aws_lambda_assume_role
module "mom_prod_aws_kms_access_policy" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-iam-policy/aws"
  version = "0.0.9"

  application           = var.application-name
  application_dl        = var.app-support-dl
  aws_create_iam_policy = local.mom_kms_access_policy["create_policy"]
  barometer-it-num      = var.apm-id
  company               = var.company
  compliance            = var.compliance
  costcenter            = var.costcenter
  description           = local.mom_kms_access_policy["description"]
  environment           = var.environment
  it-department         = var.business-division
  layer                 = var.PatchWindow
  name                  = local.mom_kms_access_policy["policy_name"]
  owner-department      = var.app-servicenow-group
  policy                = data.local_file.mom_prod_kms_access_policy.content
  resource-type         = local.mom_prod_iam_resource["resource_type"]
}

// Module for S3 Read/Write access policy to be attached for following IAM Role --> mom_prod_aws_lambda_assume_role
module "mom_prod_aws_s3_read_write_access_policy" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-iam-policy/aws"
  version = "0.0.9"

  application           = var.application-name
  application_dl        = var.app-support-dl
  aws_create_iam_policy = local.mom_s3_read_write_access_policy["create_policy"]
  barometer-it-num      = var.apm-id
  company               = var.company
  compliance            = var.compliance
  costcenter            = var.costcenter
  description           = local.mom_s3_read_write_access_policy["description"]
  environment           = var.environment
  it-department         = var.business-division
  layer                 = var.PatchWindow
  name                  = local.mom_s3_read_write_access_policy["policy_name"]
  owner-department      = var.app-servicenow-group
  policy                = data.local_file.mom_prod_s3_read_write_access_policy.content
  resource-type         = local.mom_prod_iam_resource["resource_type"]
}

// Module for AWS Transcribe access policy to be attached for following IAM Role --> mom_prod_aws_lambda_assume_role
module "mom_prod_aws_transcribe_access_policy" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-iam-policy/aws"
  version = "0.0.9"

  application           = var.application-name
  application_dl        = var.app-support-dl
  aws_create_iam_policy = local.mom_transcribe_access_policy["create_policy"]
  barometer-it-num      = var.apm-id
  company               = var.company
  compliance            = var.compliance
  costcenter            = var.costcenter
  description           = local.mom_transcribe_access_policy["description"]
  environment           = var.environment
  it-department         = var.business-division
  layer                 = var.PatchWindow
  name                  = local.mom_transcribe_access_policy["policy_name"]
  owner-department      = var.app-servicenow-group
  policy                = data.local_file.mom_prod_transcribe_access_policy.content
  resource-type         = local.mom_prod_iam_resource["resource_type"]
}

// Module for Cloudwatch Put/Get policy to be assumed by the following IAM role --> mom_prod_aws_api_gateway_assume_role
module "mom_prod_aws_api_gw_cloud_watch_access_policy" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-iam-policy/aws"
  version = "0.0.9"

  application           = var.application-name
  application_dl        = var.app-support-dl
  aws_create_iam_policy = local.mom_api_gw_cloudwatch_access_policy["create_policy"]
  barometer-it-num      = var.apm-id
  company               = var.company
  compliance            = var.compliance
  costcenter            = var.costcenter
  description           = local.mom_api_gw_cloudwatch_access_policy["description"]
  environment           = var.environment
  it-department         = var.business-division
  layer                 = var.PatchWindow
  name                  = local.mom_api_gw_cloudwatch_access_policy["policy_name"]
  owner-department      = var.app-servicenow-group
  policy                = data.local_file.mom_prod_api_gw_cloudwatch_access_policy.content
  resource-type         = local.mom_prod_iam_resource["resource_type"]
}

// Module for IAM Role to be assummed by MOM - PROD Lambda functions for Transcribe/S3/VPC access
module "mom_prod_aws_lambda_assume_role" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-iam-role/aws"
  version = "0.1.3"

  application               = var.application-name
  application_dl            = var.app-support-dl
  assume_role_service_names = local.mom_lambda_assume_role["service_names"]
  barometer-it-num          = var.apm-id
  company                   = var.company
  compliance                = var.compliance
  costcenter                = var.costcenter
  aws_create_iam_role       = local.mom_lambda_assume_role["create_role"]
  environment               = var.environment
  force_detach_policies     = local.mom_lambda_assume_role["force_detach_policies"]
  iam_role_name             = local.mom_lambda_assume_role["role_name"]
  it-department             = var.business-division
  layer                     = var.PatchWindow
  max_session_duration      = local.mom_lambda_assume_role["max_session_duration"]
  owner-department          = var.app-servicenow-group
  resource-type             = local.mom_prod_iam_resource["resource_type"]
  role_description          = local.mom_lambda_assume_role["description"]
  managed_policy_arns = ["${module.mom_prod_aws_kms_access_policy.arn}",
    "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    "arn:aws:iam::aws:policy/AWSLambda_FullAccess",
    "arn:aws:iam::aws:policy/AmazonTranscribeFullAccess",
    "arn:aws:iam::aws:policy/AmazonAPIGatewayInvokeFullAccess",
    "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
  ]
}

// Module for IAM Role to be assumed by API Gateway for S3/SSM/CloudWatch Access
module "mom_prod_aws_api_gateway_assume_role" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-iam-role/aws"
  version = "0.1.3"

  application               = var.application-name
  application_dl            = var.app-support-dl
  assume_role_service_names = local.mom_api_gw_assume_role["service_names"]
  barometer-it-num          = var.apm-id
  company                   = var.company
  compliance                = var.compliance
  costcenter                = var.costcenter
  aws_create_iam_role       = local.mom_api_gw_assume_role["create_role"]
  environment               = var.environment
  force_detach_policies     = local.mom_api_gw_assume_role["force_detach_policies"]
  iam_role_name             = local.mom_api_gw_assume_role["role_name"]
  it-department             = var.business-division
  layer                     = var.PatchWindow
  max_session_duration      = local.mom_api_gw_assume_role["max_session_duration"]
  owner-department          = var.app-servicenow-group
  resource-type             = local.mom_prod_iam_resource["resource_type"]
  role_description          = local.mom_api_gw_assume_role["description"]
  managed_policy_arns       = ["arn:aws:iam::aws:policy/CloudWatchFullAccess"]
}