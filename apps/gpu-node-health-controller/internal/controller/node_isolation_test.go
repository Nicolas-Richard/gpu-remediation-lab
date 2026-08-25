package controller

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
)

func TestIsolateNodeAddsTaintAndRecordsOwnership(t *testing.T) {
	node := &corev1.Node{}

	result := isolateNode(node)

	if !result.Changed || !result.TaintAdded {
		t.Fatalf("isolation result = %#v, want changed taint", result)
	}
	if !hasDegradedTaint(node) {
		t.Fatalf("node did not receive degraded taint: %#v", node.Spec.Taints)
	}
	if node.Annotations[ManagedAnnotation] != "true" {
		t.Fatalf("managed annotation = %q, want true", node.Annotations[ManagedAnnotation])
	}
}

func TestIsolateNodeDoesNotClaimPreExistingTaint(t *testing.T) {
	node := &corev1.Node{Spec: corev1.NodeSpec{Taints: []corev1.Taint{degradedTaint}}}

	result := isolateNode(node)

	if result.Changed || result.TaintAdded {
		t.Fatalf("isolation result = %#v, want no change", result)
	}
	if _, found := node.Annotations[ManagedAnnotation]; found {
		t.Fatal("isolation claimed ownership of a pre-existing taint")
	}
}
