#Please consider to comment this file if you upload the code to Bitbucket
terraform {
  backend "remote" {
    hostname     = "cps-terraform.anthem.com"
    organization = "DIG"

    workspaces {
      name = "awsapm1010437-prod"
    }
  }
}
