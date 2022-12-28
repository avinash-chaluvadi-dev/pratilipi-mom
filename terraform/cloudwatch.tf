module "terraform_aws_cwl_subscription_filter1" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-cwl-subscription-filter/aws"
  version = "0.0.2"
  depends_on = [module.mom_prod_aws_mail_processor]

  destination_arn = module.mom_prod_aws_splunk_forwarder.function_arn
  filter_pattern = ""
  log_group_name = "/aws/lambda/mom-batch-master-prod"
  name = "prod-splunk-logs-filter-1"
}


module "terraform_aws_cwl_subscription_filter2" {
  source  = "cps-terraform.anthem.com/DIG/terraform-aws-cwl-subscription-filter/aws"
  version = "0.0.2"
  depends_on = [module.mom_prod_aws_audio_playback]
  destination_arn = module.mom_prod_aws_splunk_forwarder.function_arn
  filter_pattern = ""
  log_group_name = "/aws/lambda/mom-audio-playback-prod"
  name = "prod-splunk-logs-filter-2"
}