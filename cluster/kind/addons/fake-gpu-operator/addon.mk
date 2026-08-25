# Fake GPU Operator provides simulated NVIDIA GPUs to the local kind workers.
FAKE_GPU_VERSION ?= 0.2.0
FAKE_GPU_CHART   ?= oci://ghcr.io/run-ai/fake-gpu-operator/fake-gpu-operator
FAKE_GPU_VALUES := cluster/kind/addons/fake-gpu-operator/values.yaml

.PHONY: _install-kind-fake-gpu-operator
_install-kind-fake-gpu-operator:
	helm upgrade --install fake-gpu-operator \
		$(FAKE_GPU_CHART) \
		--kube-context $(KUBE_CONTEXT) \
		--namespace gpu-operator --create-namespace \
		--version $(FAKE_GPU_VERSION) \
		-f $(FAKE_GPU_VALUES) \
		--wait --timeout 180s
	kubectl --context $(KUBE_CONTEXT) wait node \
		-l '!node-role.kubernetes.io/control-plane' \
		--for=jsonpath='{.status.allocatable.nvidia\.com/gpu}'=2 \
		--timeout=90s
