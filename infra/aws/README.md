# AWS GPU validation

This runbook deploys a private EKS environment with six `g6.xlarge` NVIDIA L4 workers and two CPU workers. Four GPUs run the SmolLM training workload and two provide replacement capacity. The environment uses the NVIDIA GPU Operator, real DCGM and `dcgm-exporter`, the repository's GPU health controller, and EFS-backed checkpoints.

The tests inject synthetic DCGM telemetry; they do not manufacture a physical GPU or driver failure. The training scenario separately pauses one DDP rank to create the application-visible stall.

## Prerequisites

- Terraform 1.7+, AWS CLI v2, Session Manager plugin, Docker with buildx, Helm, and kubectl.
- AWS permissions to create VPC, EKS, EC2, IAM, ECR, EFS, and CloudWatch resources.
- At least 24 On-Demand G/VT vCPUs available in `us-east-1`.

The EKS endpoint and EC2 instances are private-only. Local access uses an SSM-managed relay with no public IP or inbound rules. On macOS, install the Session Manager plugin with:

```bash
brew install --cask session-manager-plugin
session-manager-plugin --version
```

## Select the AWS account

Authenticate with any AWS credential source and set the expected account and region:

```bash
export AWS_PROFILE=your-aws-profile  # optional
export AWS_REGION=us-east-1
export AWS_EXPECTED_ACCOUNT_ID=123456789012
aws sts get-caller-identity
```

Every AWS Make target verifies that the active account matches `AWS_EXPECTED_ACCOUNT_ID` and that the region is `us-east-1` before making changes.

## Provision

```bash
cp infra/aws/terraform/terraform.tfvars.example infra/aws/terraform/terraform.tfvars
# Set the Owner tag in terraform.tfvars.

make aws-tf-plan
terraform -chdir=infra/aws/terraform show aws.tfplan
make aws-apply
```

Terraform creates the network, EKS cluster, GPU and CPU node groups, ECR repositories, and encrypted EFS checkpoint storage.

## Connect and deploy

Keep the SSM tunnel running in one terminal:

```bash
make aws-tunnel
```

In another terminal, configure the kubeconfig and deploy the GPU Operator and controller:

```bash
make aws-kubeconfig
make aws-deploy
```

The tunnel forwards the local Kubernetes context to the private EKS endpoint. `aws-deploy` builds and pushes the controller image, installs the real NVIDIA GPU/DCGM stack, and deploys the controller with `--health-source=dcgm`.

## Validate XID remediation

```bash
make test-aws-dcgm-injection-unit
make test-aws-dcgm-injection
```

The end-to-end test injects XID 79 into DCGM field 230 on a physical GPU node and verifies the controller-owned taint, targeted eviction, replacement on another GPU node, and recovery after an explicit zero value. The cleanup path clears the injected value and deletes the canary.

## Validate distributed training recovery

```bash
make test-aws-training-recovery-unit
make test-aws-training-recovery
```

This test runs four CUDA/NCCL SmolLM ranks, pauses one worker, confirms that training stalls while its pod remains `Running`, injects XID 79 on that node, and verifies that all ranks restart from the same EFS checkpoint before making new progress.

Save the complete event stream as JSON Lines or show verbose events with:

```bash
make test-aws-training-recovery AWS_TRAINING_EVENTS_JSONL=/tmp/aws-training-recovery.jsonl
make test-aws-training-recovery AWS_TRAINING_VERBOSE=1
```

See the [experiment record](../../docs/experiments/smollm_stalled_rank_xid_recovery_2026-08-24.md) for the measured transcript and evidence boundaries.

## Tear down

The tests remove their workloads but retain the infrastructure. Review the target account and Terraform outputs before destroying it:

```bash
terraform -chdir=infra/aws/terraform output
make aws-destroy
```

Six on-demand GPU workers, three NAT gateways, and EFS continue accruing cost until teardown completes.
