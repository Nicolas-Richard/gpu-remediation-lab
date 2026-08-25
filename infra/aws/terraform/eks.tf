resource "aws_cloudwatch_log_group" "eks" {
  name              = "/aws/eks/${local.cluster_name}/cluster"
  retention_in_days = 14
}

resource "aws_eks_cluster" "this" {
  name     = local.cluster_name
  role_arn = aws_iam_role.eks_cluster.arn
  version  = var.kubernetes_version

  enabled_cluster_log_types = [
    "api",
    "audit",
    "authenticator",
    "controllerManager",
    "scheduler",
  ]

  access_config {
    authentication_mode                         = "API_AND_CONFIG_MAP"
    bootstrap_cluster_creator_admin_permissions = true
  }

  vpc_config {
    endpoint_private_access = true
    endpoint_public_access  = false
    subnet_ids              = aws_subnet.private[*].id
  }

  lifecycle {
    precondition {
      condition     = data.aws_region.current.region == "us-east-1"
      error_message = "The AWS GPU gate must run in us-east-1; set AWS_REGION=us-east-1."
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.eks,
    aws_iam_role_policy_attachment.eks_cluster,
  ]
}

resource "aws_eks_node_group" "system" {
  cluster_name    = aws_eks_cluster.this.name
  node_group_name = "system"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = aws_subnet.private[*].id
  version         = var.kubernetes_version

  ami_type       = "AL2023_x86_64_STANDARD"
  capacity_type  = "ON_DEMAND"
  disk_size      = 40
  instance_types = [var.system_instance_type]

  labels = {
    "gpu-orch.dev/node-pool" = "system"
  }

  scaling_config {
    desired_size = var.system_node_count
    min_size     = var.system_node_count
    max_size     = var.system_node_count
  }

  update_config {
    max_unavailable = 1
  }

  depends_on = [aws_iam_role_policy_attachment.eks_node]
}

resource "aws_eks_node_group" "gpu" {
  cluster_name    = aws_eks_cluster.this.name
  node_group_name = "gpu"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = aws_subnet.private[*].id
  version         = var.kubernetes_version

  ami_type       = "AL2023_x86_64_NVIDIA"
  capacity_type  = "ON_DEMAND"
  disk_size      = 80
  instance_types = [var.gpu_instance_type]

  labels = {
    "gpu-orch.dev/node-pool" = "gpu"
  }

  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  scaling_config {
    desired_size = var.gpu_node_count
    min_size     = var.gpu_node_count
    max_size     = var.gpu_node_count
  }

  update_config {
    max_unavailable = 1
  }

  depends_on = [aws_iam_role_policy_attachment.eks_node]
}

data "aws_ssm_parameter" "al2023_ami" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

resource "aws_security_group" "admin" {
  name_prefix = "${local.cluster_name}-admin-"
  description = "No-ingress SSM relay for the private EKS API"
  vpc_id      = aws_vpc.this.id

  tags = {
    Name = "${local.cluster_name}-admin"
  }
}

resource "aws_vpc_security_group_egress_rule" "admin_https" {
  security_group_id = aws_security_group.admin.id
  description       = "HTTPS to SSM services and the private EKS API"
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 443
  to_port           = 443
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "eks_admin_https" {
  security_group_id            = aws_eks_cluster.this.vpc_config[0].cluster_security_group_id
  referenced_security_group_id = aws_security_group.admin.id
  description                  = "Private Kubernetes API access from the SSM relay"
  from_port                    = 443
  to_port                      = 443
  ip_protocol                  = "tcp"
}

resource "aws_instance" "admin" {
  ami                         = data.aws_ssm_parameter.al2023_ami.value
  instance_type               = var.admin_instance_type
  subnet_id                   = aws_subnet.private[0].id
  associate_public_ip_address = false
  iam_instance_profile        = aws_iam_instance_profile.admin.name
  vpc_security_group_ids      = [aws_security_group.admin.id]

  root_block_device {
    encrypted   = true
    volume_size = 12
    volume_type = "gp3"
  }

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  tags = {
    Name = "${local.cluster_name}-admin"
  }

  depends_on = [
    aws_iam_role_policy_attachment.admin_ssm,
    aws_route_table_association.private,
    aws_vpc_security_group_ingress_rule.eks_admin_https,
  ]
}
