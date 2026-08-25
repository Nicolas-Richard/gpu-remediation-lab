package source

import (
	"context"
	"fmt"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"

	"hackweek/gpu-node-health-controller/internal/health"
)

const (
	SyntheticFaultAnnotation    = "gpu-orch.dev/injected-fault"
	SyntheticDeviceAnnotation   = "gpu-orch.dev/injected-fault-device"
	SyntheticRecoveryAnnotation = "gpu-orch.dev/recovery-confirmed-at"
)

// Annotation retains deterministic local fault injection while using the same
// observation contract as real metric sources.
type Annotation struct {
	Now func() time.Time
}

func (s *Annotation) Observe(_ context.Context, node *corev1.Node) health.Observation {
	now := time.Now
	if s.Now != nil {
		now = s.Now
	}
	observedAt := now().UTC()
	fault := strings.TrimSpace(node.Annotations[SyntheticFaultAnnotation])
	device := strings.TrimSpace(node.Annotations[SyntheticDeviceAnnotation])
	if fault != "" {
		return health.Observation{
			State:      health.StateDegraded,
			Source:     "synthetic-annotation",
			Reason:     fault,
			Device:     device,
			ObservedAt: observedAt,
		}
	}

	recovery := strings.TrimSpace(node.Annotations[SyntheticRecoveryAnnotation])
	if recovery != "" {
		confirmedAt, err := time.Parse(time.RFC3339Nano, recovery)
		if err != nil {
			return health.Observation{
				State:      health.StateUnknown,
				Source:     "synthetic-annotation",
				Reason:     fmt.Sprintf("invalid recovery confirmation timestamp: %v", err),
				Device:     device,
				ObservedAt: observedAt,
			}
		}
		return health.Observation{
			State:      health.StateHealthy,
			Source:     "synthetic-annotation",
			Reason:     "recovery-confirmed",
			Device:     device,
			ObservedAt: confirmedAt.UTC(),
		}
	}

	return health.Observation{
		State:      health.StateUnknown,
		Source:     "synthetic-annotation",
		Reason:     "no synthetic observation",
		ObservedAt: observedAt,
	}
}
