AWS_TF_DIR           ?= infra/aws/terraform
AWS_TFVARS           ?= terraform.tfvars
AWS_KUBE_CONTEXT     ?= gpu-orch-aws
AWS_TUNNEL_PORT      ?= 8443
AWS_TARGET_REGION    := us-east-1
AWS_EXPECTED_ACCOUNT_ID ?=
AWS_IMAGE_TAG        := $(shell git rev-parse --short HEAD)-$(shell date -u +%Y%m%d%H%M%S)
AWS_TRAINING_BASE     ?= cluster/aws/manifests/distributed-training-base.yaml
AWS_TRAINING_EVENTS_JSONL ?=
AWS_TRAINING_EVENTS_ARG = $(if $(AWS_TRAINING_EVENTS_JSONL),--events-jsonl="$(AWS_TRAINING_EVENTS_JSONL)",)
AWS_TRAINING_VERBOSE ?=
AWS_TRAINING_VERBOSE_ARG = $(if $(filter 1 true yes,$(AWS_TRAINING_VERBOSE)),--verbose,)

.PHONY: aws-tf-init aws-tf-plan aws-apply aws-destroy aws-kubeconfig aws-tunnel \
	aws-build-push-controller aws-build-push-training aws-bootstrap aws-deploy \
	aws-training-bootstrap aws-deploy-training \
	test-aws-dcgm-injection-unit test-aws-dcgm-injection \
	test-aws-training-recovery-unit test-aws-training-recovery _check-aws-target

_check-aws-target:
	@case "$(AWS_EXPECTED_ACCOUNT_ID)" in \
		[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]) ;; \
		*) echo "Set AWS_EXPECTED_ACCOUNT_ID to the 12-digit target AWS account ID"; exit 1 ;; \
	esac
	@test "$(AWS_REGION)" = "$(AWS_TARGET_REGION)" || \
		(echo "Set AWS_REGION=$(AWS_TARGET_REGION) before using AWS targets"; exit 1)
	@actual_account="$$(aws sts get-caller-identity --query Account --output text)"; \
		test -n "$$actual_account" || \
		(echo "AWS session is unavailable; authenticate before using AWS targets"; exit 1); \
		test "$$actual_account" = "$(AWS_EXPECTED_ACCOUNT_ID)" || \
		(echo "Active AWS account $$actual_account does not match AWS_EXPECTED_ACCOUNT_ID=$(AWS_EXPECTED_ACCOUNT_ID)"; exit 1)

aws-tf-init: _check-aws-target
	terraform -chdir=$(AWS_TF_DIR) init

aws-tf-plan: aws-tf-init
	@test -f "$(AWS_TF_DIR)/$(AWS_TFVARS)" || \
		(echo "Missing $(AWS_TF_DIR)/$(AWS_TFVARS); copy terraform.tfvars.example first"; exit 1)
	terraform -chdir=$(AWS_TF_DIR) plan -var-file=$(AWS_TFVARS) -out=aws.tfplan

aws-apply: aws-tf-init
	@test -f "$(AWS_TF_DIR)/aws.tfplan" || \
		(echo "Missing $(AWS_TF_DIR)/aws.tfplan; run make aws-tf-plan and review it first"; exit 1)
	terraform -chdir=$(AWS_TF_DIR) apply aws.tfplan

aws-destroy: aws-tf-init
	@test -f "$(AWS_TF_DIR)/$(AWS_TFVARS)" || \
		(echo "Missing $(AWS_TF_DIR)/$(AWS_TFVARS); refusing to destroy"; exit 1)
	terraform -chdir=$(AWS_TF_DIR) destroy -var-file=$(AWS_TFVARS)

aws-kubeconfig: _check-aws-target
	sh infra/aws/scripts/configure-private-kubeconfig.sh \
		$(AWS_TF_DIR) $(AWS_KUBE_CONTEXT) $(AWS_TUNNEL_PORT)

aws-tunnel: _check-aws-target
	@command -v session-manager-plugin >/dev/null || \
		(echo "AWS Session Manager plugin is required"; exit 1)
	sh infra/aws/scripts/start-eks-tunnel.sh $(AWS_TF_DIR) $(AWS_TUNNEL_PORT)

aws-build-push-controller: _check-aws-target _check-docker
	@repository="$$(terraform -chdir=$(AWS_TF_DIR) output -raw controller_repository_url)"; \
	region="$$(terraform -chdir=$(AWS_TF_DIR) output -raw aws_region)"; \
	aws ecr get-login-password --region "$$region" | \
		docker login --username AWS --password-stdin "$${repository%%/*}"; \
	docker buildx build --platform linux/amd64 --push \
		--tag "$$repository:$(AWS_IMAGE_TAG)" \
		apps/gpu-node-health-controller/

aws-build-push-training: _check-aws-target _check-docker
	@repository="$$(terraform -chdir=$(AWS_TF_DIR) output -raw training_repository_url)"; \
	region="$$(terraform -chdir=$(AWS_TF_DIR) output -raw aws_region)"; \
	aws ecr get-login-password --region "$$region" | \
		docker login --username AWS --password-stdin "$${repository%%/*}"; \
	docker buildx build --platform linux/amd64 --push \
		--tag "$$repository:$(AWS_IMAGE_TAG)" \
		apps/smollm-training-workload/

aws-bootstrap: _install-aws-gpu-operator
	kubectl --context $(AWS_KUBE_CONTEXT) apply \
		-f cluster/aws/manifests/gpu-node-health-controller.yaml
	@repository="$$(terraform -chdir=$(AWS_TF_DIR) output -raw controller_repository_url)"; \
	kubectl --context $(AWS_KUBE_CONTEXT) --namespace gpu-node-health-system set image \
		deployment/gpu-node-health-controller \
		controller="$$repository:$(AWS_IMAGE_TAG)"
	kubectl --context $(AWS_KUBE_CONTEXT) --namespace gpu-node-health-system rollout status \
		deployment/gpu-node-health-controller --timeout=5m

aws-deploy: aws-build-push-controller
	$(MAKE) aws-bootstrap AWS_IMAGE_TAG=$(AWS_IMAGE_TAG)

aws-training-bootstrap: _install-aws-kubeflow-trainer
	@efs_dns="$$(terraform -chdir=$(AWS_TF_DIR) output -raw checkpoint_efs_dns_name)"; \
	sed "s|__EFS_DNS_NAME__|$$efs_dns|g" $(AWS_TRAINING_BASE) | \
		kubectl --context $(AWS_KUBE_CONTEXT) apply -f -
	kubectl --context $(AWS_KUBE_CONTEXT) --namespace gpu-training \
		wait --for=jsonpath='{.status.phase}'=Bound \
		persistentvolumeclaim/aws-training-checkpoints --timeout=3m

aws-deploy-training: aws-build-push-training aws-training-bootstrap
	@echo "AWS CUDA training image and EFS-backed Trainer runtime are ready."

test-aws-dcgm-injection-unit:
	PYTHONPYCACHEPREFIX=/tmp/hackweek-python-cache \
		python3 -m unittest discover -s tests/recovery/aws_dcgm/unit -v

test-aws-dcgm-injection: _check-aws-target
	python3 -m tests.recovery.aws_dcgm --context=$(AWS_KUBE_CONTEXT)

test-aws-training-recovery-unit:
	PYTHONPYCACHEPREFIX=/tmp/hackweek-python-cache \
		python3 -m unittest discover -s tests/recovery/aws_training/unit -v

test-aws-training-recovery: aws-build-push-training aws-training-bootstrap
	@repository="$$(terraform -chdir=$(AWS_TF_DIR) output -raw training_repository_url)"; \
	python3 -m tests.recovery.aws_training \
		--context=$(AWS_KUBE_CONTEXT) \
		--image="$$repository:$(AWS_IMAGE_TAG)" \
		$(AWS_TRAINING_EVENTS_ARG) $(AWS_TRAINING_VERBOSE_ARG)
