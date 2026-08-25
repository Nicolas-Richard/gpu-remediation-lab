.DEFAULT_GOAL := help

include versions.mk
include cluster/kind/addons/fake-gpu-operator/addon.mk
include cluster/kind/addons/kubeflow-trainer/addon.mk
include cluster/aws/addons/gpu-operator/addon.mk
include cluster/aws/addons/kubeflow-trainer/addon.mk
include infra/aws/aws-workflow.mk

CHECKPOINT_DIR := /tmp/gpu-orch-checkpoints
KIND_CONFIG    := cluster/kind/config.yaml
KUBE_CONTEXT   := kind-$(CLUSTER_NAME)
CONTROLLER_HEALTH_SOURCE ?= dcgm-simulator

.PHONY: help demo-up demo-down demo-reset \
	test-training-resume test-gpu-node-health-controller test-dcgm-metrics-simulator \
	test-fault-recovery-unit test-fault-recovery test-dcgm-fault-recovery \
	test-aws-dcgm-injection-unit test-aws-dcgm-injection \
	_check-docker _build-local-training-image \
	_build-gpu-node-health-controller-image _build-dcgm-metrics-simulator-image \
	_reset-training-state \
	_delete-training-job _remove-legacy-remediation-controller

help:
	@echo "  make demo-up           Bring up the local demo environment"
	@echo "  make demo-reset        Reset checkpoints and redeploy the demo job"
	@echo "  make demo-down         Tear down the local demo environment"
	@echo "  make test-training-resume  Test the local DDP/checkpoint workload in isolation"
	@echo "  make test-gpu-node-health-controller  Test the Go node health controller"
	@echo "  make test-dcgm-metrics-simulator  Test the Python DCGM metrics simulator"
	@echo "  make test-fault-recovery-unit  Test fault-recovery harness helpers"
	@echo "  make test-fault-recovery  Run annotation-driven E2E recovery (requires demo-up)"
	@echo "  make test-dcgm-fault-recovery  Run DCGM-driven E2E recovery (requires demo-up)"
	@echo "  make aws-apply          Create the us-east-1 GPU validation environment from an approved plan"
	@echo "  make aws-tunnel         Forward localhost:8443 to the private EKS API through SSM"
	@echo "  make aws-destroy        Destroy the AWS GPU validation environment"
	@echo "  make aws-deploy         Push the controller and install the real GPU/DCGM stack"
	@echo "  make test-aws-dcgm-injection  Run field-230 injection, eviction, and recovery on AWS"
	@echo "  make aws-deploy-training  Install Trainer and push the CUDA training image"
	@echo "  make test-aws-training-recovery  Prove EFS checkpoint recovery with real GPUs"

demo-up: _create-cluster _install-cluster-addons _load-local-training-image \
	_load-gpu-node-health-controller-image _load-dcgm-metrics-simulator-image \
	_deploy-gpu-node-health-controller _deploy-dcgm-metrics-simulator \
	_submit-training-job
	@echo "Demo up. Try: make test-dcgm-fault-recovery"

demo-down: _delete-cluster

demo-reset: _check _reset-training-state _load-local-training-image _submit-training-job
	kubectl --context $(KUBE_CONTEXT) -n training wait \
		--for=create job/demo-train-node-0 \
		--timeout=180s
	kubectl --context $(KUBE_CONTEXT) -n training wait pod \
		--for=condition=Ready \
		-l jobset.sigs.k8s.io/jobset-name=demo-train \
		--timeout=180s
	@echo "Demo reset. Try: make test-fault-recovery"

_reset-training-state: _delete-training-job
	@echo "Clearing demo checkpoints from $(CHECKPOINT_DIR)"
	@test "$(CHECKPOINT_DIR)" = "/tmp/gpu-orch-checkpoints" || \
		(echo "Refusing to clear unexpected checkpoint directory"; exit 1)
	mkdir -p "$(CHECKPOINT_DIR)"
	find "$(CHECKPOINT_DIR)" \
		-mindepth 1 -maxdepth 1 -type f \
		\( -name '*.pt' -o -name '.*.tmp' \) -delete

_delete-training-job:
	kubectl --context $(KUBE_CONTEXT) -n training delete \
		trainjob/demo-train \
		--ignore-not-found \
		--cascade=foreground
	kubectl --context $(KUBE_CONTEXT) -n training delete pods,services \
		-l jobset.sigs.k8s.io/jobset-name=demo-train \
		--ignore-not-found \
		--wait=true

test-training-resume: _build-local-training-image
	docker run --rm \
		-v "$(CURDIR)/apps/local-training-workload/tests:/tests:ro" \
		$(LOCAL_TRAINING_IMAGE) \
		python /tests/test_ddp_smoke.py -v

test-gpu-node-health-controller: _build-gpu-node-health-controller-image
	@echo "Go GPU node health controller tests passed during the image build."

test-dcgm-metrics-simulator:
	PYTHONPYCACHEPREFIX=/tmp/hackweek-python-cache \
		python3 -m unittest discover -s apps/dcgm-metrics-simulator/tests -v

test-fault-recovery-unit:
	PYTHONPYCACHEPREFIX=/tmp/hackweek-python-cache \
		python3 -m unittest discover -s tests/recovery/common/unit -v
	PYTHONPYCACHEPREFIX=/tmp/hackweek-python-cache \
		python3 -m unittest discover -s tests/recovery/kind/unit -v

test-fault-recovery: override CONTROLLER_HEALTH_SOURCE := annotation
test-fault-recovery: _check _load-gpu-node-health-controller-image \
	_deploy-gpu-node-health-controller _reset-training-state _load-local-training-image
	python3 -m tests.recovery.kind \
		--context="$(KUBE_CONTEXT)"

test-dcgm-fault-recovery: override CONTROLLER_HEALTH_SOURCE := dcgm-simulator
test-dcgm-fault-recovery: _check _load-gpu-node-health-controller-image \
	_load-dcgm-metrics-simulator-image _deploy-gpu-node-health-controller \
	_deploy-dcgm-metrics-simulator _reset-training-state _load-local-training-image
	python3 -m tests.recovery.kind \
		--context="$(KUBE_CONTEXT)" \
		--health-source=dcgm-simulator

# --- internal building blocks ------------------------------------------------

_check-docker:
	@command -v docker >/dev/null || (echo "docker missing"; exit 1)
	@docker info >/dev/null 2>&1 || (echo "Docker Desktop not running"; exit 1)

_check: _check-docker
	@command -v kind >/dev/null || (echo "kind missing"; exit 1)
	@command -v kubectl >/dev/null || (echo "kubectl missing"; exit 1)
	@command -v helm >/dev/null || (echo "helm missing"; exit 1)

_create-cluster: _check
	mkdir -p $(CHECKPOINT_DIR)
	kind create cluster \
		--name $(CLUSTER_NAME) \
		--image $(KIND_NODE_IMAGE) \
		--config $(KIND_CONFIG)
	kubectl --context $(KUBE_CONTEXT) wait --for=condition=Ready nodes --all --timeout=180s

_delete-cluster:
	kind delete cluster --name $(CLUSTER_NAME) || true

_install-cluster-addons: _install-kind-fake-gpu-operator _install-kind-kubeflow-trainer

_build-local-training-image: _check-docker
	docker build -t $(LOCAL_TRAINING_IMAGE) apps/local-training-workload/

_build-gpu-node-health-controller-image: _check-docker
	docker build -t $(GPU_NODE_HEALTH_CONTROLLER_IMAGE) \
		apps/gpu-node-health-controller/

_build-dcgm-metrics-simulator-image: _check-docker
	docker build -t $(DCGM_METRICS_SIMULATOR_IMAGE) \
		apps/dcgm-metrics-simulator/

_load-local-training-image: _build-local-training-image
	kind load docker-image $(LOCAL_TRAINING_IMAGE) --name $(CLUSTER_NAME)

_load-gpu-node-health-controller-image: _build-gpu-node-health-controller-image
	kind load docker-image $(GPU_NODE_HEALTH_CONTROLLER_IMAGE) \
		--name $(CLUSTER_NAME)

_load-dcgm-metrics-simulator-image: _build-dcgm-metrics-simulator-image
	kind load docker-image $(DCGM_METRICS_SIMULATOR_IMAGE) \
		--name $(CLUSTER_NAME)

_remove-legacy-remediation-controller:
	@kubectl --context $(KUBE_CONTEXT) delete namespace remediation-system \
		--ignore-not-found --wait=true
	@kubectl --context $(KUBE_CONTEXT) delete clusterrole,clusterrolebinding \
		remediation-controller --ignore-not-found

_deploy-gpu-node-health-controller: _remove-legacy-remediation-controller
	kubectl --context $(KUBE_CONTEXT) apply \
		-f apps/gpu-node-health-controller/manifests/controller.yaml
	kubectl --context $(KUBE_CONTEXT) -n gpu-node-health-system set env \
		deployment/gpu-node-health-controller \
		HEALTH_SOURCE=$(CONTROLLER_HEALTH_SOURCE)
	kubectl --context $(KUBE_CONTEXT) -n gpu-node-health-system rollout restart \
		deployment/gpu-node-health-controller
	kubectl --context $(KUBE_CONTEXT) -n gpu-node-health-system rollout status \
		deployment/gpu-node-health-controller --timeout=180s

_deploy-dcgm-metrics-simulator: _deploy-gpu-node-health-controller
	kubectl --context $(KUBE_CONTEXT) apply \
		-f apps/dcgm-metrics-simulator/manifests/daemonset.yaml
	kubectl --context $(KUBE_CONTEXT) -n gpu-node-health-system rollout status \
		daemonset/dcgm-metrics-simulator --timeout=180s

_submit-training-job:
	kubectl --context $(KUBE_CONTEXT) apply \
		-f apps/local-training-workload/manifests/demo-job.yaml
