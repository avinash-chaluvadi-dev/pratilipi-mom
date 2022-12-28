// Module for Eventbrige Rule which triggers aws resources once in a while
module "mom_prod_aws_mail_processor_scheduler" {compliance
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-eventbridge-rule/aws"
  version = "0.0.4"

  application         = var.application-name
  application_dl      = var.app-support-dl
  barometer-it-num    = var.apm-id
  company             = var.company
  compliance          = var.compliance
  costcenter          = var.costcenter
  description         = var.mom_prod_event_bridge_resource["mom_batch_master_scheduler"]["description"]
  environment         = var.environment
  it-department       = var.business-division
  is_enabled          = var.mom_prod_event_bridge_resource["mom_batch_master_scheduler"]["is_enabled"]
  layer               = var.PatchWindow
  name                = var.mom_prod_event_bridge_resource["mom_batch_master_scheduler"]["name"]
  owner-department    = var.app-servicenow-group
  resource-type       = var.mom_prod_event_bridge_resource["resource_type"]
  schedule_expression = var.mom_prod_event_bridge_resource["mom_batch_master_scheduler"]["schedule_expression"]
}

// Module for creating a target to mom-batch-master-prod from abobe eventbridge rule as principal
module "mom_prod_aws_mail_processor_scheduler_target" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-eventbridge-target/aws"
  version = "0.0.4"

  batch_target                   = null
  create_cloudwatch_event_target = true
  ecs_target                     = null
  event_rule_name                = module.mom_prod_aws_mail_processor_scheduler.id
  input                          = null
  input_path                     = null
  input_transformer              = null
  kinesis_target                 = null
  role_arn                       = null
  run_command_targets            = null
  sqs_target                     = null
  target_arn                     = module.mom_prod_aws_mail_processor.function_arn
  target_id                      = var.mom_prod_event_bridge_resource["mom_batch_master_scheduler"]["target"]["id"]
}