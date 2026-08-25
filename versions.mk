# Project versions shared across environments — override on the CLI if needed.
# Environment-specific dependency versions live with their add-on recipes.

CLUSTER_NAME    ?= gpu-orch
KIND_NODE_IMAGE ?= kindest/node:v1.34.0

KUBEFLOW_TRAINER_VERSION   ?= 2.2.1
KUBEFLOW_TRAINER_CHART     ?= oci://ghcr.io/kubeflow/charts/kubeflow-trainer
KUBEFLOW_TRAINER_RELEASE   ?= kubeflow-trainer
KUBEFLOW_TRAINER_NAMESPACE ?= kubeflow

LOCAL_TRAINING_IMAGE ?= local-training-workload:local
GPU_NODE_HEALTH_CONTROLLER_IMAGE ?= hackweek-gpu-node-health-controller:local
DCGM_METRICS_SIMULATOR_IMAGE ?= hackweek-dcgm-metrics-simulator:local
