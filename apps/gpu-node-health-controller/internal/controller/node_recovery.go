package controller

import corev1 "k8s.io/api/core/v1"

// nodeRecoveryResult tells the reconciler which externally visible effects
// were produced by a recovery mutation.
type nodeRecoveryResult struct {
	Changed      bool
	TaintRemoved bool
}

// recoverNode performs the complete Kubernetes mutation for node recovery: it
// removes only the exact taint owned by this controller and then releases the
// ownership marker. Unmanaged nodes are never modified.
func recoverNode(node *corev1.Node) nodeRecoveryResult {
	if node.Annotations[ManagedAnnotation] != "true" {
		return nodeRecoveryResult{}
	}

	taintRemoved := removeDegradedTaint(node)
	delete(node.Annotations, ManagedAnnotation)
	return nodeRecoveryResult{Changed: true, TaintRemoved: taintRemoved}
}

// removeDegradedTaint removes only the controller's exact isolation taint while
// preserving every unrelated taint on the node.
func removeDegradedTaint(node *corev1.Node) bool {
	taints := node.Spec.Taints[:0]
	removed := false
	for _, taint := range node.Spec.Taints {
		if isDegradedTaint(taint) {
			removed = true
			continue
		}
		taints = append(taints, taint)
	}
	node.Spec.Taints = taints
	return removed
}
