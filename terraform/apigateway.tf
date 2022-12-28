# local block to read nested values from mom_prod_api_gateway_resource map object
locals {
  rest_api                                 = var.mom_prod_api_gateway_resource["mom_s3_audio_playback_prod"]
  rest_api_resource                        = local.rest_api["resource"]
  rest_api_deployment                      = local.rest_api["deployment"]
  rest_api_get_method                      = local.rest_api["method"]["playback_method"]
  rest_api_stage                           = local.rest_api_deployment["mom_prod_stage"]
  rest_api_logs_settings                   = local.rest_api_stage["logs_settings"]
  rest_api_get_method_response             = local.rest_api_get_method["response"]
  rest_api_get_method_integration_response = local.rest_api_get_method["integration"]["response"]


  // Local variables correspond to MOM PROD API gateway arn--> mom-s3-audio-playback-prod
  get_resource = local.rest_api_resource["path_part"]
  get_method   = module.mom_prod_aws_s3_audio_playback_api_method.http_method
  rest_api_arn = "${module.mom_prod_aws_s3_audio_playback_api.execution_arn}/*/*/*"
}

// Module for Rest API Gateway endpoint creation -> mom-s3-audio-playback-prod
module "mom_prod_aws_s3_audio_playback_api" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-api-gateway-rest-api/aws"
  version = "0.0.3"

  application                  = var.application-name
  application_dl               = var.app-support-dl
  barometer-it-num             = var.apm-id
  company                      = var.company
  compliance                   = var.compliance
  costcenter                   = var.costcenter
  environment                  = var.environment
  it-department                = var.business-division
  layer                        = var.PatchWindow
  owner-department             = var.app-servicenow-group
  policy                       = null
  resource-type                = var.mom_prod_api_gateway_resource["resource_type"]
  api_binary_media_types       = local.rest_api["binary_media_types"]
  api_description              = local.rest_api["description"]
  api_minimum_compression_size = local.rest_api["minimum_compression_size"]
  api_name                     = local.rest_api["name"]
}

// Module for API Gateway child resource on -> mom-s3-audio-playback-prod
module "mom_prod_aws_s3_audio_playback_api_resource" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-api-gateway-resource/aws"
  version = "0.0.3"

  rest_api_id = module.mom_prod_aws_s3_audio_playback_api.id
  parent_id   = module.mom_prod_aws_s3_audio_playback_api.root_resource_id
  path_part   = local.rest_api_resource["path_part"]
}

// Module for API Gateway GET method on resource corresponding to -> mom-s3-audio-playback-prod
module "mom_prod_aws_s3_audio_playback_api_method" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-api-gateway-method/aws"
  version = "0.0.3"

  rest_api_id          = module.mom_prod_aws_s3_audio_playback_api.id
  resource_id          = module.mom_prod_aws_s3_audio_playback_api_resource.id
  http_method          = local.rest_api_resource["playback_method"]
  authorization        = local.rest_api_get_method["authorization"]
  authorization_scopes = local.rest_api_get_method["authorization_scopes"]
}

// Module for API Gateway GET method response on resource corresponding to -> mom-s3-audio-playback-prod
module "mom_prod_aws_s3_audio_playback_api_method_response" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-api-gateway-method-response/aws"
  version = "0.0.1"
  depends_on = [
    module.mom_prod_aws_s3_audio_playback_api,
    module.mom_prod_aws_s3_audio_playback_api_resource,
    module.mom_prod_aws_s3_audio_playback_api_method
  ]

  rest_api_id         = module.mom_prod_aws_s3_audio_playback_api.id
  resource_id         = module.mom_prod_aws_s3_audio_playback_api_resource.id
  http_method         = module.mom_prod_aws_s3_audio_playback_api_method.http_method
  status_code         = local.rest_api_get_method_response["status_code"]
  response_models     = local.rest_api_get_method_response["models"]
  response_parameters = local.rest_api_get_method_response["parameters"]
}

// Module for API Gateway GET method lambda integration on resource -> mom-s3-audio-playback-prod
module "mom_prod_aws_s3_audio_playback_api_integration" {
  source     = "cps-terraform.anthem.com/DIG/terraform-aws-api-gateway-integration/aws"
  version    = "0.0.2"
  depends_on = [module.mom_prod_aws_audio_playback]

  rest_api_id             = module.mom_prod_aws_s3_audio_playback_api.id
  resource_id             = module.mom_prod_aws_s3_audio_playback_api_resource.id
  http_method             = module.mom_prod_aws_s3_audio_playback_api_method.http_method
  integration_http_method = local.rest_api_get_method["integration"]["http_method"]
  integration_type        = local.rest_api_get_method["integration"]["type"]
  uri                     = module.mom_prod_aws_audio_playback.invoke_arn
  connection_type         = null
}

// Module for API Gateway GET method lambda integration response on resource -> mom-s3-audio-playback-prod
module "mom_prod_aws_s3_audio_playback_api_integration_response" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-api-gateway-integration-response/aws"
  version = "0.0.1"

  depends_on = [
    module.mom_prod_aws_s3_audio_playback_api,
    module.mom_prod_aws_s3_audio_playback_api_method,
    module.mom_prod_aws_s3_audio_playback_api_resource,
    module.mom_prod_aws_s3_audio_playback_api_integration
  ]
  rest_api_id         = module.mom_prod_aws_s3_audio_playback_api.id
  resource_id         = module.mom_prod_aws_s3_audio_playback_api_resource.id
  http_method         = module.mom_prod_aws_s3_audio_playback_api_method.http_method
  content_handling    = local.rest_api_get_method_integration_response["content_handling"]
  status_code         = local.rest_api_get_method_integration_response["status_code"]
  response_parameters = local.rest_api_get_method_integration_response["parameters"]
}

// Module for Rest API Gateway endpoint deployment -> mom-s3-audio-playback-prod
module "mom_prod_aws_s3_audio_playback_api_deployment" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-api-gateway-deployment/aws"
  version = "0.0.1"
  depends_on = [
    module.mom_prod_aws_s3_audio_playback_api_method,
    module.mom_prod_aws_s3_audio_playback_api_integration
  ]

  rest_api_id            = module.mom_prod_aws_s3_audio_playback_api.id
  deployment_description = local.rest_api_deployment["description"]
  triggers = {
    redeployment = sha1(jsonencode([
      module.mom_prod_aws_s3_audio_playback_api.id,
      module.mom_prod_aws_s3_audio_playback_api_resource.id,
      module.mom_prod_aws_s3_audio_playback_api_method.http_method
      ])
    )
  }
}

// Module for Rest API Gateway endpoint deployment stage -> mom-s3-audio-playback-prod
module "mom_prod_aws_s3_audio_playback_api_stage" {
  source     = "cps-terraform.anthem.com/DIG/terraform-aws-api-gateway-stage/aws"
  version    = "0.0.2"
  depends_on = [module.mom_prod_aws_s3_audio_playback_api_deployment]

  application           = var.application-name
  application_dl        = var.app-support-dl
  barometer-it-num      = var.apm-id
  company               = var.company
  compliance            = var.compliance
  costcenter            = var.costcenter
  environment           = var.environment
  it-department         = var.business-division
  layer                 = var.PatchWindow
  owner-department      = var.app-servicenow-group
  resource-type         = var.mom_prod_api_gateway_resource["resource_type"]
  rest_api_id           = module.mom_prod_aws_s3_audio_playback_api.id
  stage_name            = local.rest_api_stage["name"]
  deployment_id         = module.mom_prod_aws_s3_audio_playback_api_deployment.id
  stage_description     = local.rest_api_stage["description"]
  cache_cluster_enabled = local.rest_api_stage["cache_cluster_enabled"]
  cache_cluster_size    = local.rest_api_stage["cache_cluster_size"]
  xray_tracing_enabled  = local.rest_api_stage["xray_tracing_enabled"]
  stage_variables       = {}
}

//Module for the API Gateway endpoint(mom-s3-audio-playback-prod) to facilitate access to CloudWatch
module "mom_prod_aws_s3_audio_playback_api_cloudwatch_permissions" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-api-gateway-account/aws"
  version = "0.0.2"

  cloudwatch_role_arn = module.mom_prod_aws_api_gateway_assume_role.iamrole_arn
}

//Module for enabling logs/tracing on API Gateway endpoint --> mom-s3-audio-playback-prod
module "mom_prod_aws_s3_audio_playback_api_cloudwatch_logs_settings" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-api-gateway-method-settings/aws"
  version = "0.0.1"
  depends_on = [
    module.mom_prod_aws_s3_audio_playback_api, 
    module.mom_prod_aws_s3_audio_playback_api_stage,
    module.mom_prod_aws_s3_audio_playback_api_method, 
    module.mom_prod_aws_s3_audio_playback_api_cloudwatch_permissions
  ]

  rest_api_id          = module.mom_prod_aws_s3_audio_playback_api.id
  method_path          = local.rest_api_logs_settings["method_path"]
  logging_level        = local.rest_api_logs_settings["logging_level"]
  caching_enabled      = local.rest_api_logs_settings["caching_enabled"]
  metrics_enabled      = local.rest_api_logs_settings["metrics_enabled"]
  data_trace_enabled   = local.rest_api_logs_settings["data_trace_enabled"]
  cache_ttl_in_seconds = local.rest_api_logs_settings["cache_ttl_in_seconds"]
  stage_name           = local.rest_api_deployment["mom_prod_stage"]["name"]
}