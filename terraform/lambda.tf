# local block to read nested values from mom_prod_lambda_resource map object
locals {
  lambda_resource              = var.mom_prod_lambda_resource
  batch_master_sc_location     = local.lambda_resource["mom_batch_master"]["source_code_location"]
  packages_sc_location         = local.lambda_resource["python_packages_layer"]["source_code_location"]
  audio_playback_sc_location   = local.lambda_resource["mom_audio_playback"]["source_code_location"]
  splunk_forwarder_sc_location = local.lambda_resource["mom_splunk_forwarder"]["source_code_location"]

  batch_master_sc_object     = local.lambda_resource["mom_batch_master"][local.batch_master_sc_location]
  packages_sc_object         = local.lambda_resource["python_packages_layer"][local.packages_sc_location]
  audio_playback_sc_object   = local.lambda_resource["mom_audio_playback"][local.audio_playback_sc_location]
  splunk_forwarder_sc_object = local.lambda_resource["mom_splunk_forwarder"][local.splunk_forwarder_sc_location]


  // Local variables correspond to lambda permissions/resource based policy
  audio_playback_permission   = local.lambda_resource["mom_audio_playback"]["resource_policy"]
  splunk_forwarder_permission = local.lambda_resource["mom_splunk_forwarder"]["resource_policy"]
  batch_master_permission     = var.mom_prod_event_bridge_resource["mom_batch_master_scheduler"]["target"]["permissions"]

  // Local variables correspond to custom generated KMS Key for lambda
  lambda_kms_key_arn           = local.lambda_resource["kms_key_arn"]
  lambda_log_prefix            = replace(var.mom_prod_cloudwatch_resource["log_group_prefix"], "{account_id}", local.account_id)
  batch_master_log_group_arn   = replace(local.lambda_log_prefix, "{function_name}", local.lambda_resource["mom_batch_master"]["name"])
  audio_playback_log_group_arn = replace(local.lambda_log_prefix, "{function_name}", local.lambda_resource["mom_audio_playback"]["name"])
}

// Module for the Mailbox processor/Audio serve lambda function layer -> mom-batch-master-prod
module "mom_prod_aws_lambda_python_layer" {
  source     = "cps-terraform.anthem.com/DIG/terraform-aws-lambda-layer-version/aws"
  version    = "0.0.1"
  depends_on = [module.mom_prod_aws_s3_bucket_python_packages_layer_object]

  layer_name           = local.lambda_resource["python_packages_layer"]["name"]
  s3_bucket            = local.packages_sc_object["bucket_name"]
  s3_key               = local.packages_sc_object["packages_key"]
  // s3_object_version    = local.packages_sc_object["object_version"]
  compatible_runtimes  = local.lambda_resource["python_packages_layer"]["runtimes"]
  description          = local.lambda_resource["python_packages_layer"]["description"]
  source_code_location = local.packages_sc_location
}

// Module for the Mailbox processor lambda function -> mom-batch-master-prod
module "mom_prod_aws_mail_processor" {
  source     = "cps-terraform.anthem.com/DIG/terraform-aws-lambda/aws"
  version    = "0.2.0"
  depends_on = [module.mom_prod_aws_s3_bucket_mail_processor_object]

  application          = var.application-name
  application_dl       = var.app-support-dl
  barometer-it-num     = var.apm-id
  company              = var.company
  compliance           = var.compliance
  costcenter           = var.costcenter
  description          = local.lambda_resource["mom_batch_master"]["description"]
  environment          = var.environment
  function_name        = local.lambda_resource["mom_batch_master"]["name"]
  handler              = local.lambda_resource["mom_batch_master"]["handler"]
  runtime              = local.lambda_resource["mom_batch_master"]["runtime"]
  timeout              = local.lambda_resource["mom_batch_master"]["timeout"]
  it-department        = var.business-division
  kms_key_arn          = local.lambda_kms_key_arn
  layer                = var.PatchWindow
  owner-department     = var.app-servicenow-group
  resource-type        = local.lambda_resource["resource_type"]
  role                 = module.mom_prod_aws_lambda_assume_role.iamrole_arn
  source_code_location = local.batch_master_sc_location
  s3_bucket            = local.batch_master_sc_object["bucket_name"]
  s3_key               = local.batch_master_sc_object["source_code_key"]
  subnet_ids           = var.mom_prod_subnet_ids
  security_group_ids   = var.mom_prod_security_group_ids
  layers               = [module.mom_prod_aws_lambda_python_layer.arn]
  lambda_environment   = local.lambda_resource["mom_batch_master"]["environment"]
}
// Module for adding resource based policy to above lambda
module "mom_prod_aws_mail_processor_resource_policy" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-lambda-permission/aws"
  version = "0.0.3"

  action                   = var.lambda_invoke_action
  create_lambda_permission = local.batch_master_permission["create_lambda_permission"]
  event_source_token       = null
  function_name            = module.mom_prod_aws_mail_processor.function_arn
  principal                = var.mom_prod_event_bridge_resource["principal"]
  qualifier                = null
  source_account           = null
  source_arn               = module.mom_prod_aws_mail_processor_scheduler.arn
  statement_id             = local.batch_master_permission["statement_id"]
  statement_id_prefix      = null
}

// Module for the Audio playback lambda function -> mom-audio-playback-prod
module "mom_prod_aws_audio_playback" {
  source     = "cps-terraform.anthem.com/DIG/terraform-aws-lambda/aws"
  version    = "0.2.0"
  depends_on = [module.mom_prod_aws_s3_bucket_audio_playback_object]

  application          = var.application-name
  application_dl       = var.app-support-dl
  barometer-it-num     = var.apm-id
  company              = var.company
  compliance           = var.compliance
  costcenter           = var.costcenter
  description          = local.lambda_resource["mom_audio_playback"]["description"]
  environment          = var.environment
  function_name        = local.lambda_resource["mom_audio_playback"]["name"]
  handler              = local.lambda_resource["mom_audio_playback"]["handler"]
  runtime              = local.lambda_resource["mom_audio_playback"]["runtime"]
  timeout              = local.lambda_resource["mom_audio_playback"]["timeout"]
  it-department        = var.business-division
  kms_key_arn          = local.lambda_kms_key_arn
  layer                = var.PatchWindow
  owner-department     = var.app-servicenow-group
  resource-type        = local.lambda_resource["resource_type"]
  role                 = module.mom_prod_aws_lambda_assume_role.iamrole_arn
  source_code_location = local.audio_playback_sc_location
  s3_bucket            = local.audio_playback_sc_object["bucket_name"]
  s3_key               = local.audio_playback_sc_object["source_code_key"]
  subnet_ids           = var.mom_prod_subnet_ids
  security_group_ids   = var.mom_prod_security_group_ids
  layers               = [module.mom_prod_aws_lambda_python_layer.arn]
  lambda_environment   = local.lambda_resource["mom_audio_playback"]["environment"]
}


module "mom_prod_aws_audio_playback_resource_policy" {
  source     = "cps-terraform.anthem.com/DIG/terraform-aws-lambda-permission/aws"
  version    = "0.0.3"
  depends_on = [module.mom_prod_aws_s3_audio_playback_api]

  action                   = var.lambda_invoke_action
  create_lambda_permission = local.audio_playback_permission["create_lambda_permission"]
  event_source_token       = null
  function_name            = module.mom_prod_aws_audio_playback.function_arn
  principal                = var.mom_prod_api_gateway_resource["principal"]
  qualifier                = null
  source_account           = null
  source_arn               = local.rest_api_arn
  statement_id             = local.audio_playback_permission["statement_id"]
  statement_id_prefix      = null
}

// Module for the cloudwatch logs forwrder to splunk -> mom-splunk-forwarder-prod
module "mom_prod_aws_splunk_forwarder" {
  source     = "cps-terraform.anthem.com/DIG/terraform-aws-lambda/aws"
  version    = "0.2.0"
  depends_on = [module.mom_prod_aws_s3_bucket_splunk_forwarder_object]

  application          = var.application-name
  application_dl       = var.app-support-dl
  barometer-it-num     = var.apm-id
  company              = var.company
  compliance           = var.compliance
  costcenter           = var.costcenter
  description          = local.lambda_resource["mom_splunk_forwarder"]["description"]
  environment          = var.environment
  function_name        = local.lambda_resource["mom_splunk_forwarder"]["name"]
  handler              = local.lambda_resource["mom_splunk_forwarder"]["handler"]
  runtime              = local.lambda_resource["mom_splunk_forwarder"]["runtime"]
  timeout              = local.lambda_resource["mom_splunk_forwarder"]["timeout"]
  memory_size          = local.lambda_resource["mom_splunk_forwarder"]["memory_size"]
  it-department        = var.business-division
  kms_key_arn          = local.lambda_kms_key_arn
  layer                = var.PatchWindow
  owner-department     = var.app-servicenow-group
  resource-type        = local.lambda_resource["resource_type"]
  role                 = module.mom_prod_aws_lambda_assume_role.iamrole_arn
  source_code_location = local.splunk_forwarder_sc_location
  s3_bucket            = local.splunk_forwarder_sc_object["bucket_name"]
  s3_key               = local.splunk_forwarder_sc_object["source_code_key"]
  subnet_ids           = var.mom_prod_subnet_ids
  security_group_ids   = var.mom_prod_security_group_ids
  lambda_environment   = local.lambda_resource["mom_splunk_forwarder"]["environment"]
  tracing_config_mode  = local.lambda_resource["mom_splunk_forwarder"]["tracing_config_mode"]
}
// Module for adding resource based policy to above lambda
module "mom_prod_aws_mail_processor_splunk_forwarder_policy" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-lambda-permission/aws"
  version = "0.0.3"

  action                   = var.lambda_invoke_action
  create_lambda_permission = local.splunk_forwarder_permission["create_lambda_permission"]
  event_source_token       = null
  function_name            = module.mom_prod_aws_splunk_forwarder.function_arn
  principal                = var.mom_prod_cloudwatch_resource["principal"]
  qualifier                = null
  source_account           = null
  source_arn               = local.batch_master_log_group_arn
  statement_id             = local.splunk_forwarder_permission["batch_master_statement_id"]
  statement_id_prefix      = null
}
module "mom_prod_aws_audio_playback_splunk_forwarder_policy" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-lambda-permission/aws"
  version = "0.0.3"

  action                   = var.lambda_invoke_action
  create_lambda_permission = local.splunk_forwarder_permission["create_lambda_permission"]
  event_source_token       = null
  function_name            = module.mom_prod_aws_splunk_forwarder.function_arn
  principal                = var.mom_prod_cloudwatch_resource["principal"]
  qualifier                = null
  source_account           = null
  source_arn               = local.audio_playback_log_group_arn
  statement_id             = local.splunk_forwarder_permission["audio_playback_statement_id"]
  statement_id_prefix      = null
}