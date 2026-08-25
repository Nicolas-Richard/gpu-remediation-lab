# Kubernetes GPU Failure Recovery

This project demonstrates automated recovery from GPU failures that do not change Kubernetes' standard node-health signals. A custom controller classifies XID telemetry from DCGM exporter, isolates the affected node, and evicts opted-in training pods. Kubernetes then recreates the distributed training group, which resumes from a shared checkpoint.

Read [Building a Kubernetes GPU Health Controller: From XID Detection to Checkpoint Recovery](docs/blog.md) for the project walkthrough, or see the [experiment record](docs/experiments/smollm_stalled_rank_xid_recovery_2026-08-24.md) for the final AWS validation.

## Components

- [`apps/gpu-node-health-controller`](apps/gpu-node-health-controller) — Go controller for XID classification, node isolation, targeted eviction, and recovery.
- [`apps/local-training-workload`](apps/local-training-workload) and [`apps/dcgm-metrics-simulator`](apps/dcgm-metrics-simulator) — CPU-only DDP and simulated XID fixtures for local testing.
- [`apps/smollm-training-workload`](apps/smollm-training-workload) — CUDA/NCCL SmolLM workload with shared checkpoint recovery.
- [`tests/recovery`](tests/recovery) — recovery scenarios for local kind and AWS.
- [`infra/aws`](infra/aws) and [`cluster`](cluster) — the EKS infrastructure and environment-specific cluster configuration.

## Run locally

Prerequisites: Docker Desktop, `kind`, `kubectl`, and `helm`.

```bash
make demo-up
make test-dcgm-fault-recovery
make demo-down
```

The local scenario uses fake GPU resources, simulated DCGM metrics, and a CPU-only distributed workload; it does not require a GPU. Run `make` to list the other test targets.

## AWS validation

The AWS scenario runs four SmolLM training ranks on NVIDIA L4 GPUs with two spare GPU nodes. It pauses one rank, verifies that the DDP group stalls while Kubernetes still reports the pod as running, injects XID 79 through DCGM, and confirms node isolation, pod replacement, and recovery from an EFS checkpoint.

See the [AWS runbook](infra/aws/README.md) for prerequisites, deployment, validation, and teardown.
