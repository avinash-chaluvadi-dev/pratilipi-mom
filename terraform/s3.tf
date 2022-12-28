# local block to read s3 bucket object values from mom_prod_s3_resource map object
locals {
  python_packages_s3_key  = local.packages_sc_object["packages_key"]
  batch_master_s3_key     = local.batch_master_sc_object["source_code_key"]
  audio_playback_s3_key   = local.audio_playback_sc_object["source_code_key"]
  splunk_forwarder_s3_key = local.splunk_forwarder_sc_object["source_code_key"]

  batch_master_source     = local.lambda_resource["mom_batch_master"]["source_location"]
  audio_playback_source   = local.lambda_resource["mom_audio_playback"]["source_location"]
  splunk_forwarder_source = local.lambda_resource["mom_splunk_forwarder"]["source_location"]
  python_packages_source  = local.lambda_resource["python_packages_layer"]["source_location"]

  // Local variables correspond to custom generated KMS Key for S3
  s3_kms_key_arn = var.mom_prod_s3_resource["kms_key_arn"]
}

//Module for MOM S3 Bucket - PROD Environment
module "mom_prod_aws_s3_bucket" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-s3/aws"
  version = "0.3.5"

  application      = var.application-name
  application_dl   = var.app-support-dl
  aws_kms_key_arn  = local.s3_kms_key_arn
  barometer-it-num = var.apm-id
  bucket           = var.mom_prod_s3_resource["bucket_name"]
  company          = var.company
  compliance       = var.compliance
  costcenter       = var.costcenter
  environment      = var.environment
  it-department    = var.business-division
  layer            = var.PatchWindow
  owner-department = var.app-servicenow-group
  resource-type    = var.mom_prod_s3_resource["resource_type"]
}


//Module for uploading source code corresponding to lambda layer to S3 bucket - PROD Environment
module "mom_prod_aws_s3_bucket_python_packages_layer_object" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-s3-bucket-object/aws"
  version = "0.1.2"

  application-name  = var.application-name
  company           = var.company
  compliance        = var.compliance
  costcenter        = var.costcenter
  business-division = var.apm-id

  bucket           = module.mom_prod_aws_s3_bucket.id
  key              = [local.python_packages_s3_key]
  kms_key_id       = local.s3_kms_key_arn
  source_file_name = local.python_packages_source
  source_hash      = filemd5(local.python_packages_source)
}

//Module for uploading source code corresponding to lambda layer into S3 bucket - PROD Environment
module "mom_prod_aws_s3_bucket_mail_processor_object" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-s3-bucket-object/aws"
  version = "0.1.2"

  application-name  = var.application-name
  company           = var.company
  compliance        = var.compliance
  costcenter        = var.costcenter
  business-division = var.apm-id

  bucket           = module.mom_prod_aws_s3_bucket.id
  key              = [local.batch_master_s3_key]
  kms_key_id       = local.s3_kms_key_arn
  source_file_name = local.batch_master_source
  source_hash      = filemd5(local.batch_master_source)
}

//Module for uploading source code corresponding to mom-batch-master into S3 bucket - PROD Environment
module "mom_prod_aws_s3_bucket_audio_playback_object" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-s3-bucket-object/aws"
  version = "0.1.2"

  application-name  = var.application-name
  company           = var.company
  compliance        = var.compliance
  costcenter        = var.costcenter
  business-division = var.apm-id

  bucket           = module.mom_prod_aws_s3_bucket.id
  key              = [local.audio_playback_s3_key]
  kms_key_id       = local.s3_kms_key_arn
  source_file_name = local.audio_playback_source
  source_hash      = filemd5(local.audio_playback_source)
}

//Module for uploading source code corresponding to lambda layer to S3 bucket - PROD Environment
module "mom_prod_aws_s3_bucket_splunk_forwarder_object" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-s3-bucket-object/aws"
  version = "0.1.2"

  application-name  = var.application-name
  company           = var.company
  compliance        = var.compliance
  costcenter        = var.costcenter
  business-division = var.apm-id

  bucket           = module.mom_prod_aws_s3_bucket.id
  key              = [local.splunk_forwarder_s3_key]
  kms_key_id       = local.s3_kms_key_arn
  source_file_name = local.splunk_forwarder_source
  source_hash      = filemd5(local.splunk_forwarder_source)
}