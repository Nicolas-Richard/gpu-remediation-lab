# Cluster software

This directory contains environment-specific Kubernetes add-ons and manifests. Infrastructure and connectivity live under [`infra`](../infra), while application source and reusable manifests live under [`apps`](../apps).

| Component | kind | AWS | Purpose |
| --- | --- | --- | --- |
| Kubeflow Trainer | [`kind/addons/kubeflow-trainer`](kind/addons/kubeflow-trainer) | [`aws/addons/kubeflow-trainer`](aws/addons/kubeflow-trainer) | Reconcile `TrainJob` resources into JobSets |
| GPU Operator | [`kind/addons/fake-gpu-operator`](kind/addons/fake-gpu-operator) | [`aws/addons/gpu-operator`](aws/addons/gpu-operator) | Provide simulated GPUs locally and the NVIDIA GPU/DCGM stack on AWS |
| GPU health controller | [`apps/gpu-node-health-controller`](../apps/gpu-node-health-controller) | [`aws/manifests/gpu-node-health-controller.yaml`](aws/manifests/gpu-node-health-controller.yaml) | Classify XIDs, isolate nodes, and evict opted-in workloads |

The add-on recipes install third-party controllers with explicit Helm commands; generated manifests and third-party source are not vendored.

```text
TrainJob -> Kubeflow Trainer -> JobSet -> JobSet controller -> Jobs -> Pods
```
