output "aws_account_id" {
  description = "AWS account containing the gate."
  value       = data.aws_caller_identity.current.account_id
}

output "aws_region" {
  description = "AWS region selected through the standard AWS environment."
  value       = data.aws_region.current.region
}

output "cluster_name" {
  description = "EKS cluster name used by aws eks update-kubeconfig."
  value       = aws_eks_cluster.this.name
}

output "cluster_endpoint" {
  description = "Private EKS API endpoint used as the SSM tunnel destination."
  value       = aws_eks_cluster.this.endpoint
}

output "cluster_endpoint_hostname" {
  description = "Private EKS API hostname used for SSM forwarding and TLS verification."
  value       = trimprefix(aws_eks_cluster.this.endpoint, "https://")
}

output "admin_instance_id" {
  description = "Private, no-ingress EC2 relay targeted by the SSM tunnel."
  value       = aws_instance.admin.id
}

output "controller_repository_url" {
  description = "ECR repository to which the amd64 controller image is pushed."
  value       = aws_ecr_repository.controller.repository_url
}

output "training_repository_url" {
  description = "ECR repository to which the CUDA distributed-training image is pushed."
  value       = aws_ecr_repository.training.repository_url
}

output "checkpoint_efs_dns_name" {
  description = "Regional EFS hostname mounted by the distributed-training checkpoint volume."
  value       = "${aws_efs_file_system.checkpoints.id}.efs.${data.aws_region.current.region}.amazonaws.com"
}

output "gpu_node_group" {
  description = "Managed GPU node group name."
  value       = aws_eks_node_group.gpu.node_group_name
}
