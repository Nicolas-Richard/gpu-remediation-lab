package controller

import (
	"context"
	"strings"

	corev1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

const (
	// TrainingJobLabel identifies the JobSet created for a Kubeflow TrainJob.
	TrainingJobLabel = "jobset.sigs.k8s.io/jobset-name"
	// AutoRemediateLabel opts an individual training pod into automatic eviction.
	AutoRemediateLabel = "gpu-orch.dev/auto-remediate"
)

// PodEvicter hides the Kubernetes Eviction API behind the one operation the
// reconciler needs. Keeping this as an interface makes controller tests simple.
type PodEvicter interface {
	// Evict requests graceful removal of one pod.
	Evict(ctx context.Context, namespace, name string) error
}

type KubernetesPodEvicter struct {
	clientset kubernetes.Interface
}

func NewKubernetesPodEvicter(clientset kubernetes.Interface) *KubernetesPodEvicter {
	return &KubernetesPodEvicter{clientset: clientset}
}

func (e *KubernetesPodEvicter) Evict(ctx context.Context, namespace, name string) error {
	return e.clientset.PolicyV1().Evictions(namespace).Evict(ctx, &policyv1.Eviction{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
	})
}

// isImpactedTrainingPod reports whether a live Kubeflow training pod is still
// assigned to the degraded node and therefore needs eviction.
func isImpactedTrainingPod(pod *corev1.Pod, nodeName string) bool {
	if pod.Spec.NodeName != nodeName || pod.DeletionTimestamp != nil {
		return false
	}
	if pod.Status.Phase == corev1.PodSucceeded || pod.Status.Phase == corev1.PodFailed {
		return false
	}
	return strings.TrimSpace(pod.Labels[TrainingJobLabel]) != "" &&
		strings.TrimSpace(pod.Labels[AutoRemediateLabel]) == "true"
}
