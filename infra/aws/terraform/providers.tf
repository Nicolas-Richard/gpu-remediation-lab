provider "aws" {
  default_tags {
    tags = local.tags
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  cluster_name = "${var.name_prefix}-dcgm-gate"
  azs          = slice(data.aws_availability_zones.available.names, 0, 3)

  tags = merge(
    {
      Project   = "kubernetes-gpu-fault-recovery"
      ManagedBy = "terraform"
      Gate      = "dcgm-injection"
    },
    var.tags,
  )
}
