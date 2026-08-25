package source

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"hackweek/gpu-node-health-controller/internal/health"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func TestEvaluateXIDMetrics(t *testing.T) {
	now := time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC)
	metric := func(value string) string {
		return "# TYPE DCGM_FI_DEV_XID_ERRORS gauge\n" +
			"DCGM_FI_DEV_XID_ERRORS{UUID=\"GPU-deadbeef\",device=\"nvidia0\"} " + value + "\n"
	}

	tests := []struct {
		name   string
		body   string
		state  health.State
		reason string
	}{
		{name: "critical XID 31", body: metric("31"), state: health.StateDegraded, reason: "critical-xid-31"},
		{name: "critical XID 43", body: metric("43"), state: health.StateDegraded, reason: "critical-xid-43"},
		{name: "critical XID 62", body: metric("62"), state: health.StateDegraded, reason: "critical-xid-62"},
		{name: "critical XID 79", body: metric("79"), state: health.StateDegraded, reason: "critical-xid-79"},
		{name: "explicit zero", body: metric("0"), state: health.StateHealthy, reason: "xid-clear"},
		{name: "missing series", body: "DCGM_FI_DEV_GPU_UTIL 0\n", state: health.StateUnknown, reason: "absent"},
		{name: "unclassified XID", body: metric("13"), state: health.StateUnknown, reason: "unclassified"},
		{name: "non-integral XID", body: metric("31.5"), state: health.StateUnknown, reason: "31.5"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := EvaluateXIDMetrics(strings.NewReader(test.body), now)
			if got.State != test.state || !strings.Contains(got.Reason, test.reason) {
				t.Fatalf("observation = %#v, want state %s reason containing %q", got, test.state, test.reason)
			}
			if !got.ObservedAt.Equal(now) {
				t.Fatalf("observedAt = %s, want %s", got.ObservedAt, now)
			}
		})
	}
}

func TestXIDPolicyTableContainsOnlyExplicitlyClassifiedValues(t *testing.T) {
	tests := []struct {
		value  float64
		found  bool
		state  health.State
		reason string
	}{
		{value: 0, found: true, state: health.StateHealthy, reason: "xid-clear"},
		{value: 31, found: true, state: health.StateDegraded, reason: "critical-xid-31"},
		{value: 43, found: true, state: health.StateDegraded, reason: "critical-xid-43"},
		{value: 62, found: true, state: health.StateDegraded, reason: "critical-xid-62"},
		{value: 79, found: true, state: health.StateDegraded, reason: "critical-xid-79"},
		{value: 13, found: false},
	}

	for _, test := range tests {
		policy, found := xidPolicies[test.value]
		if found != test.found {
			t.Fatalf("XID %g policy found = %t, want %t", test.value, found, test.found)
		}
		if found && (policy.State != test.state || policy.Reason != test.reason) {
			t.Fatalf(
				"XID %g policy = %#v, want state %s reason %q",
				test.value,
				policy,
				test.state,
				test.reason,
			)
		}
	}
}

func TestDCGMScrapesExporterAssignedToNode(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	exporter := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "nvidia-dcgm-exporter-a",
			Namespace: "gpu-operator",
			Labels: map[string]string{
				"app":       "nvidia-dcgm-exporter",
				"component": "status-exporter",
			},
		},
		Spec: corev1.PodSpec{NodeName: "node-a"},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			PodIP: "127.0.0.1",
			Conditions: []corev1.PodCondition{{
				Type:   corev1.PodReady,
				Status: corev1.ConditionTrue,
			}},
		},
	}
	reader := fake.NewClientBuilder().WithScheme(scheme).WithObjects(exporter).Build()
	source := &DCGM{
		PodReader: reader,
		Namespace: "gpu-operator",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body: io.NopCloser(strings.NewReader(
					"# TYPE DCGM_FI_DEV_XID_ERRORS gauge\n" +
						"DCGM_FI_DEV_XID_ERRORS{UUID=\"GPU-node-a\"} 79\n",
				)),
				Header: make(http.Header),
			}, nil
		})},
		EndpointForPod: func(*corev1.Pod) string {
			return "http://dcgm.test/metrics"
		},
	}

	got := source.Observe(context.Background(), &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}})
	if got.State != health.StateDegraded || got.Device != "GPU-node-a" {
		t.Fatalf("observation = %#v, want degraded GPU-node-a", got)
	}
}

func TestDCGMUsesConfiguredSimulatorSelectorAndSourceName(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	exporter := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "dcgm-metrics-simulator-a",
			Namespace: "gpu-node-health-system",
			Labels: map[string]string{
				"gpu-orch.dev/health-exporter": "dcgm-simulator",
			},
		},
		Spec: corev1.PodSpec{NodeName: "node-a"},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			PodIP: "127.0.0.1",
			Conditions: []corev1.PodCondition{{
				Type:   corev1.PodReady,
				Status: corev1.ConditionTrue,
			}},
		},
	}
	reader := fake.NewClientBuilder().WithScheme(scheme).WithObjects(exporter).Build()
	source := &DCGM{
		PodReader:  reader,
		Namespace:  "gpu-node-health-system",
		SourceName: "dcgm-simulator",
		Labels: map[string]string{
			"gpu-orch.dev/health-exporter": "dcgm-simulator",
		},
		HTTPClient: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body: io.NopCloser(strings.NewReader(
					"# TYPE DCGM_FI_DEV_XID_ERRORS gauge\n" +
						"DCGM_FI_DEV_XID_ERRORS{UUID=\"GPU-node-a\"} 0\n",
				)),
				Header: make(http.Header),
			}, nil
		})},
		EndpointForPod: func(*corev1.Pod) string {
			return "http://simulator.test/metrics"
		},
	}

	got := source.Observe(context.Background(), &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}})
	if got.State != health.StateHealthy || got.Source != "dcgm-simulator" {
		t.Fatalf("observation = %#v, want healthy dcgm-simulator", got)
	}
}
