VAULT_ADDR           = ""
lambda_invoke_action = "lambda:InvokeFunction"
mom_prod_bucket_name = "mom-voicemail-bucket-prod"

mom_prod_s3_resource = {
  "resource_type" = "s3"
  "bucket_name"   = "mom-voicemail-bucket-prod"
  "kms_key_arn"   = "arn:aws:kms:us-east-1:454134248856:key/653c8024-72dd-44b7-a5bf-2d72f33b23fb"
}

mom_prod_iam_resource = {
  "resource_type" = "iam"
  "role_policies" = "mom_prod_role_policies"
  "mom_ec2_modify_attribute_policy" = {
    "create_policy" = true
    "file_name"     = "mom_prod_ec2_modify_attribute.json"
    "policy_name"   = "mom-prod-ec2-modify-attribute-policy"
    "description"   = "EC2 ModifyInstanceAttribute policy to be attached to 'mom-voicemail-role' for MOM - PROD"
  }
  "mom_kms_access_policy" = {
    "create_policy" = true
    "file_name"     = "mom_prod_kms_access.json"
    "policy_name"   = "mom-prod-kms-access-policy"
    "description"   = "KMS full access policy to be attached to 'mom-voicemail-role' for MOM - PROD"
  }
  "mom_s3_read_write_access_policy" = {
    "create_policy" = true
    "file_name"     = "mom_prod_s3_read_write.json"
    "policy_name"   = "mom-prod-s3-read-write-access-policy"
    "description"   = "S3 Read/Write policy to be attached to 'mom-voicemail-role' for MOM - PROD"
  }
  "mom_transcribe_access_policy" = {
    "create_policy" = true
    "file_name"     = "mom_prod_transcribe_access.json"
    "policy_name"   = "mom-prod-transcribe-access-policy"
    "description"   = "AWS Transcribe policy to be attached to 'mom-voicemail-role' for MOM - PROD"
  }
  "mom_api_gw_cloudwatch_access_policy" = {
    "create_policy" = true
    "file_name"     = "mom_prod_api_gw_cloudwatch.json"
    "policy_name"   = "mom-prod-api-gw-cloudwatch-access-policy"
    "description"   = "Cloudwatch Put/Get events policy to be attached to 'mom-voicemail-role' for MOM - PROD"
  }
  "mom_lambda_assume_role" = {
    "create_role"           = true
    "max_session_duration"  = 25000
    "force_detach_policies" = "false"
    "role_name"             = "mom-prod-voicemail-role"
    "service_names"         = ["lambda.amazonaws.com", "s3.amazonaws.com"]
    "description"           = "IAM Role to be used for the MOM - PROD Lambda functions"
  }
  "mom_api_gw_assume_role" = {
    "create_role"           = true
    "max_session_duration"  = 25000
    "force_detach_policies" = "false"
    "role_name"             = "mom-prod-api-gw-cloudwatch-role"
    "description"           = "IAM Role to be used for the MOM - PROD AWS API Gateway resources"
    "service_names"         = ["apigateway.amazonaws.com", "lambda.amazonaws.com", "cloudwatch.amazonaws.com"]
  }
}

mom_prod_lambda_resource = {
  "resource_type" = "lambda"
  "kms_key_arn"   = "arn:aws:kms:us-east-1:454134248856:key/15301321-7f73-41fc-9153-76fc8e0f26e9"
  "python_packages_layer" = {
    "source_code_location" = "s3"
    "name"                 = "mom-python-layer-prod"
    "runtimes"             = ["python3.7", "python3.8", "python3.9"]
    "description"          = "Python package dependencies for lambda functions"
    "source_location"      = "mom_prod_s3_terraform/lambdas/layer/mom-python-layer-prod.zip"
    "s3" = {
      "bucket_name"    = "mom-voicemail-bucket-prod"
      "object_version" = ""
      "packages_key"   = "terraform/lambdas/layer/mom-python-layer-prod.zip"
    }
  }
  "mom_batch_master" = {
    "timeout"              = 900
    "source_code_location" = "s3"
    "runtime"              = "python3.8"
    "name"                 = "mom-batch-master-prod"
    "handler"              = "mom_batch.mailbox_processor_handler"
    "description"          = "Pratilipi MOM Voice Mail Processor Lambda"
    "source_location"      = "mom_prod_s3_terraform/lambdas/source_code/mom-batch-master-prod.zip"
    "s3" = {
      "bucket_name"     = "mom-voicemail-bucket-prod"
      "source_code_key" = "terraform/lambdas/source-code/mom-batch-master-prod.zip"
    }
    "environment" = {
      "variables" = {
        "CONFIG_URL"      = "https://ms.blue.digiproducts.ps.awsdns.internal.das/python-util-config-server/vmt-mom-gateway-prod,eksprodblue.json"
        "DECRYPT_URL"     = "https://ms.blue.digiproducts.ps.awsdns.internal.das/python-util-config-server/decrypt"
        "CONFIG_URL_CERT" = ""
      }
    }
  }
  "mom_audio_playback" = {
    "timeout"              = 300
    "source_code_location" = "s3"
    "runtime"              = "python3.8"
    "name"                 = "mom-audio-playback-prod"
    "handler"              = "audio_playback.audio_playback_handler"
    "description"          = "Responsible for playing/rendering audio from S3 into UI"
    "source_location"      = "mom_prod_s3_terraform/lambdas/source_code/mom-audio-playback-prod.zip"
    "s3" = {
      "bucket_name"     = "mom-voicemail-bucket-prod"
      "source_code_key" = "terraform/lambdas/source-code/mom-audio-playback-prod.zip"
    }
    "environment" = {
      "variables" = {
        "CONFIG_URL"      = "https://ms.blue.digiproducts.ps.awsdns.internal.das/python-util-config-server/vmt-mom-gateway-prod,eksprodblue.json"
        "DECRYPT_URL"     = "https://ms.blue.digiproducts.ps.awsdns.internal.das/python-util-config-server/decrypt"
        "CONFIG_URL_CERT" = ""
      }
    }
    "resource_policy" = {
      "create_lambda_permission" = true
      "statement_id"             = "mom-s3-audio-playback-lambda-target-permission-prod"
    }
  }
  "mom_splunk_forwarder" = {
    "timeout"              = 600
    "memory_size"          = 1024
    "source_code_location" = "s3"
    "tracing_config_mode"  = "Active"
    "runtime"              = "nodejs14.x"
    "handler"              = "splunk/index.handler"
    "name"                 = "mom-splunk-fowarder-prod"
    "description"          = "Responsible for forwarding cloudwatch logs into specific splunk index"
    "source_location"      = "mom_prod_s3_terraform/lambdas/source_code/mom-splunk-forwarder-prod.zip"
    "s3" = {
      "bucket_name"     = "mom-voicemail-bucket-prod"
      "source_code_key" = "terraform/lambdas/source-code/mom-splunk-forwarder-prod.zip"
    }
    "environment" = {
      "variables" = {
        "SPLUNK_HEC_TOKEN" = "80b4d861-534e-4aac-83b7-a6118c3e3680"
        "SPLUNK_HEC_URL"   = "https://hec.net-log.internal.das/services/collector/event"
      }
    }
    "resource_policy" = {
      "create_lambda_permission"    = true
      "batch_master_statement_id"   = "mom-batch-master-splunk-lambda-target-permission-prod"
      "audio_playback_statement_id" = "mom-audio-playback-splunk-lambda-target-permission-prod"
    }
  }
}

mom_prod_kms_resource = {
  "create_kms_key" = true
  "multi_region"   = true
  "resource_type"  = "kms"
  "description"    = "KMS keys for Lambda function and S3 Bucket"
  "service_names"  = { "lambda_function" : "lambda", "s3_bucket" : "s3" }
}

mom_prod_event_bridge_resource = {
  "resource_type" = "CWE"
  "principal"     = "events.amazonaws.com"
  "mom_batch_master_scheduler" = {
    "is_enabled"          = true
    "schedule_expression" = "rate(10 minutes)"
    "name"                = "mom-batch-master-scheduler-prod"
    "description"         = "Triggers the mom-batch-master Lambda function every 10 minutes periodically"
    "target" = {
      "id" = "mom-batch-master-scheduler-lambda-target-prod"
      "permissions" = {
        "create_lambda_permission" = true
        "statement_id"             = "mom-batch-master-scheduler-lambda-target-permission-prod"
      }
    }

  }
}

mom_prod_api_gateway_resource = {
  "resource_type" = "api"
  "principal"     = "apigateway.amazonaws.com"
  "mom_s3_audio_playback_prod" = {
    "minimum_compression_size" = "-1"
    "binary_media_types"       = ["*/*"]
    "name"                     = "mom-s3-audio-playback-prod"
    "description"              = "Responsible for audio playback from S3 on UI - PROD"
    "resource" = {
      "playback_method" = "GET"
      "path_part"       = "mom-s3-audio-playback-api-resource"
    }
    "method" = {
      "playback_method" = {
        "authorization_scopes" = []
        "authorization"        = "NONE"
        "response" = {
          "status_code" = "200"
          "models"      = { "application/json" = "Empty" }
          "parameters" = {
            "method.response.header.Access-Control-Allow-Origin" = false
          }
        }
        "integration" = {
          "type"        = "AWS_PROXY"
          "http_method" = "POST"
          "response" = {
            "status_code"      = "200"
            "content_handling" = "CONVERT_TO_TEXT"
            "models"           = { "application/json" = "Empty" }
            "parameters" = {
              "method.response.header.Access-Control-Allow-Origin" = "'*'"
            }
          }
        }
      }
    }
    "deployment" = {
      "description" = "MOM S3 audio-playback Rest API Deployment"
      "mom_prod_stage" = {
        "name"                  = "mom-prod-stage"
        "cache_cluster_size"    = "0.5"
        "xray_tracing_enabled"  = true
        "cache_cluster_enabled" = false
        "description"           = "MOM S3 audio-playback Rest API PROD Deployment stage"
        "logs_settings" = {
          "method_path"          = "*/*"
          "logging_level"        = "INFO"
          "caching_enabled"      = true
          "metrics_enabled"      = true
          "data_trace_enabled"   = true
          "cache_ttl_in_seconds" = "300"
        }
      }
    }
  }
}

mom_prod_cloudwatch_resource = {
  "principal"        = "logs.us-east-1.amazonaws.com"
  "log_group_prefix" = "arn:aws:logs:us-east-1:{account_id}:log-group:/aws/lambda/{function_name}:*"

}