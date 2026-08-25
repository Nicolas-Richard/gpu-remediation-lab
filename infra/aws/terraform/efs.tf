resource "aws_efs_file_system" "checkpoints" {
  encrypted        = true
  performance_mode = "generalPurpose"
  throughput_mode  = "bursting"

  tags = {
    Name = "${local.cluster_name}-checkpoints"
  }
}

resource "aws_security_group" "efs" {
  name_prefix = "${local.cluster_name}-efs-"
  description = "NFS access to the distributed-training checkpoint filesystem"
  vpc_id      = aws_vpc.this.id

  tags = {
    Name = "${local.cluster_name}-efs"
  }
}

resource "aws_vpc_security_group_ingress_rule" "efs_nodes" {
  security_group_id            = aws_security_group.efs.id
  referenced_security_group_id = aws_eks_cluster.this.vpc_config[0].cluster_security_group_id
  description                  = "NFS from EKS worker nodes"
  from_port                    = 2049
  to_port                      = 2049
  ip_protocol                  = "tcp"
}

resource "aws_efs_mount_target" "checkpoints" {
  count = length(aws_subnet.private)

  file_system_id  = aws_efs_file_system.checkpoints.id
  subnet_id       = aws_subnet.private[count.index].id
  security_groups = [aws_security_group.efs.id]
}
