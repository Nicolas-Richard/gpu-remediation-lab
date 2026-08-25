package controller

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestRecoverNodeRemovesOwnedTaintAndPreservesOtherTaints(t *testing.T) {
	externalTaint := corev1.Taint{
		Key:    DegradedTaintKey,
		Value:  "external-value",
		Effect: corev1.TaintEffectNoSchedule,
	}
	unrelatedTaint := corev1.Taint{
		Key:    "example.com/keep",
		Value:  "true",
		Effect: corev1.TaintEffectNoSchedule,
	}
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{ManagedAnnotation: "true"}},
		Spec: corev1.NodeSpec{Taints: []corev1.Taint{
			degradedTaint,
			externalTaint,
			unrelatedTaint,
		}},
	}

	result := recoverNode(node)

	if !result.Changed || !result.TaintRemoved {
		t.Fatalf("recovery result = %#v, want removed taint", result)
	}
	if _, found := node.Annotations[ManagedAnnotation]; found {
		t.Fatal("managed annotation was not removed")
	}
	if len(node.Spec.Taints) != 2 || node.Spec.Taints[0] != externalTaint || node.Spec.Taints[1] != unrelatedTaint {
		t.Fatalf("remaining taints = %#v, want external and unrelated taints", node.Spec.Taints)
	}
}

func TestRecoverNodeDoesNotModifyUnmanagedNode(t *testing.T) {
	node := &corev1.Node{Spec: corev1.NodeSpec{Taints: []corev1.Taint{degradedTaint}}}

	result := recoverNode(node)

	if result.Changed || result.TaintRemoved {
		t.Fatalf("recovery result = %#v, want no change", result)
	}
	if !hasDegradedTaint(node) {
		t.Fatal("recovery removed an unmanaged taint")
	}
}

func TestRecoverNodeReleasesOwnershipWhenTaintIsAlreadyAbsent(t *testing.T) {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{ManagedAnnotation: "true"}},
	}

	result := recoverNode(node)

	if !result.Changed || result.TaintRemoved {
		t.Fatalf("recovery result = %#v, want annotation-only change", result)
	}
	if _, found := node.Annotations[ManagedAnnotation]; found {
		t.Fatal("stale managed annotation was not removed")
	}
}
