# Kubeflow Trainer manages TrainJob resources and installs the JobSet controller.
.PHONY: _install-aws-kubeflow-trainer
_install-aws-kubeflow-trainer: aws-kubeconfig
	helm upgrade --install $(KUBEFLOW_TRAINER_RELEASE) $(KUBEFLOW_TRAINER_CHART) \
		--kube-context $(AWS_KUBE_CONTEXT) \
		--version $(KUBEFLOW_TRAINER_VERSION) \
		--namespace $(KUBEFLOW_TRAINER_NAMESPACE) \
		--create-namespace \
		--wait \
		--timeout 10m
	# The chart's upgrade hook rotates its self-signed webhook secret without
	# changing the Deployment pod template. Restart so the server certificate
	# and the newly patched admission CA bundle come from the same generation.
	kubectl --context $(AWS_KUBE_CONTEXT) --namespace $(KUBEFLOW_TRAINER_NAMESPACE) \
		rollout restart deployment/kubeflow-trainer-controller-manager
	kubectl --context $(AWS_KUBE_CONTEXT) --namespace $(KUBEFLOW_TRAINER_NAMESPACE) \
		rollout status deployment/jobset-controller --timeout=5m
	kubectl --context $(AWS_KUBE_CONTEXT) --namespace $(KUBEFLOW_TRAINER_NAMESPACE) \
		rollout status deployment/kubeflow-trainer-controller-manager --timeout=5m
