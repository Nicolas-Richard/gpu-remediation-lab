variable "name_prefix" {
  description = "Prefix applied to AWS resources."
  type        = string
  default     = "gpu-orch"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,20}$", var.name_prefix))
    error_message = "name_prefix must be 2-21 lowercase letters, numbers, or hyphens and start with a letter."
  }
}

variable "kubernetes_version" {
  description = "Amazon EKS control-plane and managed-node Kubernetes version."
  type        = string
  default     = "1.34"
}

variable "gpu_instance_type" {
  description = "Single-GPU EC2 instance type used by each validation worker."
  type        = string
  default     = "g6.xlarge"
}

variable "gpu_node_count" {
  description = "Number of on-demand GPU workers; four training ranks plus two failover targets."
  type        = number
  default     = 6

  validation {
    condition     = var.gpu_node_count >= 6 && var.gpu_node_count <= 8
    error_message = "gpu_node_count must be between 6 and 8."
  }
}

variable "system_instance_type" {
  description = "EC2 instance type for non-GPU system workloads and the controller."
  type        = string
  default     = "m7i.large"
}

variable "system_node_count" {
  description = "Number of on-demand system workers."
  type        = number
  default     = 2

  validation {
    condition     = var.system_node_count >= 1 && var.system_node_count <= 4
    error_message = "system_node_count must be between 1 and 4."
  }
}

variable "admin_instance_type" {
  description = "Small private EC2 instance used only as an SSM port-forward relay."
  type        = string
  default     = "t3.micro"
}

variable "tags" {
  description = "Additional tags merged into every supported AWS resource."
  type        = map(string)
  default     = {}
}
