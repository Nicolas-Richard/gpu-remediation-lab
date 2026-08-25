package controller

import corev1 "k8s.io/api/core/v1"

const (
	// ManagedAnnotation records that this controller owns the degraded taint and
	// may therefore remove it after confirmed recovery.
	ManagedAnnotation = "gpu-orch.dev/node-health-managed"
	// DegradedTaintKey and DegradedTaintValue identify the NoSchedule taint that
	// keeps replacement workloads away from an unhealthy node.
	DegradedTaintKey   = "gpu-orch.dev/hardware-degraded"
	DegradedTaintValue = "true"
)

// degradedTaint is the exact Kubernetes isolation state this controller owns.
var degradedTaint = corev1.Taint{
	Key:    DegradedTaintKey,
	Value:  DegradedTaintValue,
	Effect: corev1.TaintEffectNoSchedule,
}

// nodeIsolationResult tells the reconciler which externally visible effects
// were produced by an isolation mutation.
type nodeIsolationResult struct {
	Changed    bool
	TaintAdded bool
}

// isolateNode performs the complete Kubernetes mutation for node isolation: it
// adds the controller's exact taint and records ownership of that taint. A
// pre-existing taint is deliberately not claimed, because the controller must
// never later remove isolation created by another actor.
func isolateNode(node *corev1.Node) nodeIsolationResult {
	if !ensureDegradedTaint(node) {
		return nodeIsolationResult{}
	}
	if node.Annotations == nil {
		node.Annotations = make(map[string]string)
	}
	node.Annotations[ManagedAnnotation] = "true"
	return nodeIsolationResult{Changed: true, TaintAdded: true}
}

// ensureDegradedTaint isolates the node from new scheduling. It does not
// rewrite a pre-existing taint with a different value.
func ensureDegradedTaint(node *corev1.Node) bool {
	for _, taint := range node.Spec.Taints {
		if isDegradedTaint(taint) {
			return false
		}
	}
	node.Spec.Taints = append(node.Spec.Taints, degradedTaint)
	return true
}

func isDegradedTaint(taint corev1.Taint) bool {
	return taint.Key == degradedTaint.Key &&
		taint.Value == degradedTaint.Value &&
		taint.Effect == degradedTaint.Effect
}
