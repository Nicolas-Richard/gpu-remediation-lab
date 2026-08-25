package source

import (
	"context"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"

	"hackweek/gpu-node-health-controller/internal/health"
)

func TestAnnotationRequiresExplicitRecovery(t *testing.T) {
	now := time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC)
	source := &Annotation{Now: func() time.Time { return now }}
	node := &corev1.Node{}

	if got := source.Observe(context.Background(), node); got.State != health.StateUnknown {
		t.Fatalf("missing annotation state = %s, want unknown", got.State)
	}
	node.Annotations = map[string]string{SyntheticRecoveryAnnotation: now.Format(time.RFC3339)}
	if got := source.Observe(context.Background(), node); got.State != health.StateHealthy || !got.ObservedAt.Equal(now) {
		t.Fatalf("confirmed recovery state = %s, want healthy", got.State)
	}
	node.Annotations[SyntheticRecoveryAnnotation] = "not-a-timestamp"
	if got := source.Observe(context.Background(), node); got.State != health.StateUnknown {
		t.Fatalf("invalid recovery state = %s, want unknown", got.State)
	}
}
