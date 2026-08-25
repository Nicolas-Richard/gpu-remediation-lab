# NVIDIA GPU Operator supplies device discovery, the device plugin, DCGM, and
# dcgm-exporter on the EKS AL2023 NVIDIA worker nodes.
AWS_GPU_OPERATOR_VERSION ?= v26.3.3
AWS_GPU_OPERATOR_CHART   ?= https://helm.ngc.nvidia.com/nvidia/charts/gpu-operator-$(AWS_GPU_OPERATOR_VERSION).tgz
AWS_GPU_OPERATOR_VALUES  := cluster/aws/addons/gpu-operator/values.yaml

.PHONY: _install-aws-gpu-operator
_install-aws-gpu-operator: aws-kubeconfig
	helm upgrade --install gpu-operator $(AWS_GPU_OPERATOR_CHART) \
		--kube-context $(AWS_KUBE_CONTEXT) \
		--namespace gpu-operator \
		--create-namespace \
		--values $(AWS_GPU_OPERATOR_VALUES) \
		--wait \
		--timeout 15m
