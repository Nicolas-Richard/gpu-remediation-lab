package source

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"hackweek/gpu-node-health-controller/internal/health"
)

const DCGMXIDMetric = "DCGM_FI_DEV_XID_ERRORS"

type xidPolicy struct {
	State  health.State
	Reason string
}

// xidPolicies is the complete set of XID values the controller can classify.
// Values absent from this table are unknown rather than implicitly healthy.
var xidPolicies = map[float64]xidPolicy{
	0:  {State: health.StateHealthy, Reason: "xid-clear"},
	31: {State: health.StateDegraded, Reason: "critical-xid-31"},
	43: {State: health.StateDegraded, Reason: "critical-xid-43"},
	62: {State: health.StateDegraded, Reason: "critical-xid-62"},
	79: {State: health.StateDegraded, Reason: "critical-xid-79"},
}

// DCGM discovers the DCGM exporter running on a node and scrapes its Prometheus
// endpoint directly, preserving the exporter Pod-to-node map.
type DCGM struct {
	PodReader      client.Reader
	Namespace      string
	Labels         map[string]string
	LabelSelector  string
	SourceName     string
	Port           int
	HTTPClient     *http.Client
	EndpointForPod func(*corev1.Pod) string
	Now            func() time.Time
}

func (s *DCGM) Observe(ctx context.Context, node *corev1.Node) health.Observation {
	now := time.Now
	if s.Now != nil {
		now = s.Now
	}
	unknown := func(reason string) health.Observation {
		return health.Observation{
			State:      health.StateUnknown,
			Source:     s.sourceName(),
			Reason:     reason,
			ObservedAt: now().UTC(),
		}
	}
	if s.PodReader == nil {
		return unknown("DCGM exporter pod reader is not configured")
	}

	pods := &corev1.PodList{}
	matchLabels := s.Labels
	if len(matchLabels) == 0 && strings.TrimSpace(s.LabelSelector) == "" {
		matchLabels = map[string]string{
			"app":       "nvidia-dcgm-exporter",
			"component": "status-exporter",
		}
	}
	labelOption := client.ListOption(client.MatchingLabels(matchLabels))
	if strings.TrimSpace(s.LabelSelector) != "" {
		selector, err := labels.Parse(strings.TrimSpace(s.LabelSelector))
		if err != nil {
			return unknown(fmt.Sprintf("parse DCGM exporter label selector: %v", err))
		}
		labelOption = client.MatchingLabelsSelector{Selector: selector}
	}
	if err := s.PodReader.List(
		ctx,
		pods,
		client.InNamespace(s.Namespace),
		labelOption,
	); err != nil {
		return unknown(fmt.Sprintf("list DCGM exporters: %v", err))
	}

	sort.Slice(pods.Items, func(i, j int) bool {
		return pods.Items[i].Name < pods.Items[j].Name
	})
	var exporter *corev1.Pod
	for i := range pods.Items {
		pod := &pods.Items[i]
		if pod.Spec.NodeName == node.Name && pod.Status.Phase == corev1.PodRunning &&
			pod.Status.PodIP != "" && podReady(pod) {
			exporter = pod
			break
		}
	}
	if exporter == nil {
		return unknown("no ready DCGM exporter on node")
	}

	port := s.Port
	if port == 0 {
		port = 9400
	}
	endpoint := fmt.Sprintf("http://%s:%d/metrics", exporter.Status.PodIP, port)
	if s.EndpointForPod != nil {
		endpoint = s.EndpointForPod(exporter)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return unknown(fmt.Sprintf("build DCGM scrape request: %v", err))
	}
	httpClient := s.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 5 * time.Second}
	}
	response, err := httpClient.Do(request)
	if err != nil {
		return unknown(fmt.Sprintf("scrape DCGM exporter: %v", err))
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return unknown(fmt.Sprintf("scrape DCGM exporter: HTTP %d", response.StatusCode))
	}

	observation := EvaluateXIDMetrics(io.LimitReader(response.Body, 4<<20), now().UTC())
	observation.Source = s.sourceName()
	return observation
}

func (s *DCGM) sourceName() string {
	if strings.TrimSpace(s.SourceName) != "" {
		return strings.TrimSpace(s.SourceName)
	}
	return "dcgm"
}

// EvaluateXIDMetrics applies the explicit XID policy table. Missing metrics and
// unclassified values are unknown, never healthy, so they cannot accidentally
// clear node isolation.
func EvaluateXIDMetrics(reader io.Reader, observedAt time.Time) health.Observation {
	unknown := func(reason string) health.Observation {
		return health.Observation{
			State:      health.StateUnknown,
			Source:     "dcgm",
			Reason:     reason,
			ObservedAt: observedAt.UTC(),
		}
	}

	metricFamilies, err := (&expfmt.TextParser{}).TextToMetricFamilies(reader)
	if err != nil {
		return unknown(fmt.Sprintf("parse DCGM metrics: %v", err))
	}
	family := metricFamilies[DCGMXIDMetric]
	if family == nil || len(family.Metric) == 0 {
		return unknown("metric " + DCGMXIDMetric + " absent")
	}

	unclassified := make([]string, 0)
	for _, metric := range family.Metric {
		value, ok := metricValue(metric)
		if !ok {
			return unknown("metric " + DCGMXIDMetric + " has no numeric value")
		}
		device := metricLabel(metric, "UUID")
		if device == "" {
			device = metricLabel(metric, "device")
		}
		policy, classified := xidPolicies[value]
		if !classified {
			unclassified = append(unclassified, fmt.Sprintf("%g", value))
			continue
		}
		if policy.State == health.StateDegraded {
			return health.Observation{
				State:      health.StateDegraded,
				Source:     "dcgm",
				Reason:     policy.Reason,
				Device:     device,
				ObservedAt: observedAt.UTC(),
			}
		}
	}
	if len(unclassified) > 0 {
		return unknown("unclassified XID values: " + strings.Join(unclassified, ","))
	}
	return health.Observation{
		State:      health.StateHealthy,
		Source:     "dcgm",
		Reason:     xidPolicies[0].Reason,
		ObservedAt: observedAt.UTC(),
	}
}

func podReady(pod *corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

func metricValue(metric *dto.Metric) (float64, bool) {
	switch {
	case metric.Gauge != nil && metric.Gauge.Value != nil:
		return metric.GetGauge().GetValue(), true
	case metric.Counter != nil && metric.Counter.Value != nil:
		return metric.GetCounter().GetValue(), true
	case metric.Untyped != nil && metric.Untyped.Value != nil:
		return metric.GetUntyped().GetValue(), true
	default:
		return 0, false
	}
}

func metricLabel(metric *dto.Metric, name string) string {
	for _, label := range metric.Label {
		if label.GetName() == name {
			return label.GetValue()
		}
	}
	return ""
}
