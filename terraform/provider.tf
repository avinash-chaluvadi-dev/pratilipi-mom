# local block to get account details from aws_caller_identity data source
locals {
  account_id = data.aws_caller_identity.mom_prod_account.account_id
}

// Vault provider block
provider "vault" {
  address = var.VAULT_ADDR
  auth_login {
    path = "auth/approle/login"
    parameters = {
      role_id   = var.APP_ROLE_ID
      secret_id = var.APP_ROLE_SECRET_ID
    }
  }
}

// Vault data block - credentials
data "vault_aws_access_credentials" "creds" {
  backend = "${var.VAULT_NAMESPACE_XYZ}/aws/${var.ACCOUNT_TYPE}"
  role    = var.ACCOUNT_TYPE
  type    = "sts"
}

// Data block to get aws account details
data "aws_caller_identity" "mom_prod_account" {}

# AWS provider block
provider "aws" {
  default_tags {
    tags = {
      apm-id               = var.apm-id
      PatchGroup           = var.PatchGroup
      PatchWindow          = var.PatchWindow
      workspace            = var.ATLAS_WORKSPACE_NAME
      app-servicenow-group = var.app-servicenow-group
    }
  }
  region     = "us-east-1"
  access_key = data.vault_aws_access_credentials.creds.access_key
  secret_key = data.vault_aws_access_credentials.creds.secret_key
  token      = data.vault_aws_access_credentials.creds.security_token
}