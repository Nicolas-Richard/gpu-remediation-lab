package source

import (
	"context"

	corev1 "k8s.io/api/core/v1"

	"hackweek/gpu-node-health-controller/internal/health"
)

// Source produces the current health observation for one Kubernetes node.
type Source interface {
	Observe(ctx context.Context, node *corev1.Node) health.Observation
}
