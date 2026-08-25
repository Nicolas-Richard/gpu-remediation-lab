# Kubeflow Trainer manages the demo's TrainJob and installs the JobSet controller.
.PHONY: _install-kind-kubeflow-trainer
_install-kind-kubeflow-trainer:
	helm upgrade --install $(KUBEFLOW_TRAINER_RELEASE) $(KUBEFLOW_TRAINER_CHART) \
		--kube-context $(KUBE_CONTEXT) \
		--version $(KUBEFLOW_TRAINER_VERSION) \
		--namespace $(KUBEFLOW_TRAINER_NAMESPACE) \
		--create-namespace \
		--wait \
		--timeout 5m
	kubectl --context $(KUBE_CONTEXT) -n $(KUBEFLOW_TRAINER_NAMESPACE) rollout status \
		deployment/jobset-controller --timeout=180s
	kubectl --context $(KUBE_CONTEXT) -n $(KUBEFLOW_TRAINER_NAMESPACE) rollout status \
		deployment/kubeflow-trainer-controller-manager --timeout=180s
