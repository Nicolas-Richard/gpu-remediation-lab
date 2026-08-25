package controller

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"hackweek/gpu-node-health-controller/internal/health"
	healthsource "hackweek/gpu-node-health-controller/internal/health/source"
)

type evictionRecord struct {
	namespace string
	name      string
}

type recordingEvicter struct {
	records []evictionRecord
	errors  map[types.NamespacedName]error
}

func (e *recordingEvicter) Evict(_ context.Context, namespace, name string) error {
	e.records = append(e.records, evictionRecord{namespace: namespace, name: name})
	return e.errors[types.NamespacedName{Namespace: namespace, Name: name}]
}

func TestReconcileTaintsNodeAndEvictsOnlyOptedInTrainingPods(t *testing.T) {
	scheme := testScheme(t)
	node := faultedNode("worker-a")
	trainingPod := podOnNode("training", "demo-worker", "worker-a", remediableLabels("demo-train"))
	optedOutPod := podOnNode(
		"training",
		"opted-out-worker",
		"worker-a",
		map[string]string{TrainingJobLabel: "other-train"},
	)
	unrelatedPod := podOnNode("default", "web", "worker-a", map[string]string{"app": "web"})
	otherNodePod := podOnNode("training", "other-worker", "worker-b", remediableLabels("other-train"))

	kubernetesClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		node,
		trainingPod,
		optedOutPod,
		unrelatedPod,
		otherNodePod,
	).Build()
	evicter := &recordingEvicter{}
	recorder := record.NewFakeRecorder(10)
	reconciler := &NodeHealthReconciler{
		Client:       kubernetesClient,
		PodReader:    kubernetesClient,
		Evicter:      evicter,
		Recorder:     recorder,
		HealthSource: &healthsource.Annotation{},
	}

	if _, err := reconciler.Reconcile(context.Background(), requestFor(node)); err != nil {
		t.Fatalf("reconcile returned an error: %v", err)
	}

	updatedNode := getNode(t, kubernetesClient, node.Name)
	if !hasDegradedTaint(updatedNode) {
		t.Fatalf("node did not receive degraded taint: %#v", updatedNode.Spec.Taints)
	}
	if got := updatedNode.Annotations[ManagedAnnotation]; got != "true" {
		t.Fatalf("managed annotation = %q, want true", got)
	}
	if len(evicter.records) != 1 || evicter.records[0].namespace != "training" || evicter.records[0].name != "demo-worker" {
		t.Fatalf("evictions = %#v, want only training/demo-worker", evicter.records)
	}

	events := readEvents(t, recorder, 2)
	joinedEvents := strings.Join(events, "\n")
	for _, expected := range []string{
		"GPUNodeHealthDegraded",
		"TrainingPodEvictionRequested",
		"node=worker-a",
		"device=unknown",
		"source=synthetic-annotation",
		"state=degraded",
		"reason=xid-79",
		"observedAt=",
		"action=",
	} {
		if !strings.Contains(joinedEvents, expected) {
			t.Fatalf("events %q do not contain %q", joinedEvents, expected)
		}
	}
}

func TestReconcileRequiresConfiguredHealthSource(t *testing.T) {
	scheme := testScheme(t)
	node := faultedNode("worker-a")
	kubernetesClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(node).Build()
	reconciler := &NodeHealthReconciler{Client: kubernetesClient}

	_, err := reconciler.Reconcile(context.Background(), requestFor(node))
	if err == nil || !strings.Contains(err.Error(), "health source is not configured") {
		t.Fatalf("reconcile error = %v, want missing health source error", err)
	}
}

func TestReconcileDoesNotClaimOrRemovePreExistingTaint(t *testing.T) {
	scheme := testScheme(t)
	node := faultedNode("worker-a")
	node.Spec.Taints = []corev1.Taint{degradedTaint}
	kubernetesClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(node).Build()
	reconciler := newTestReconciler(kubernetesClient, &recordingEvicter{})

	if _, err := reconciler.Reconcile(context.Background(), requestFor(node)); err != nil {
		t.Fatalf("fault reconcile returned an error: %v", err)
	}
	updatedNode := getNode(t, kubernetesClient, node.Name)
	if _, found := updatedNode.Annotations[ManagedAnnotation]; found {
		t.Fatal("controller claimed ownership of a pre-existing degraded taint")
	}

	delete(updatedNode.Annotations, InjectedFaultAnnotation)
	if err := kubernetesClient.Update(context.Background(), updatedNode); err != nil {
		t.Fatal(err)
	}
	if _, err := reconciler.Reconcile(context.Background(), requestFor(node)); err != nil {
		t.Fatalf("cleanup reconcile returned an error: %v", err)
	}
	updatedNode = getNode(t, kubernetesClient, node.Name)
	if !hasDegradedTaint(updatedNode) {
		t.Fatalf("controller removed a pre-existing degraded taint: %#v", updatedNode.Spec.Taints)
	}
}

func TestRepeatedReconcileDoesNotDuplicateTaintOrOwnershipEvent(t *testing.T) {
	scheme := testScheme(t)
	node := faultedNode("worker-a")
	kubernetesClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(node).Build()
	recorder := record.NewFakeRecorder(10)
	reconciler := newTestReconciler(kubernetesClient, &recordingEvicter{})
	reconciler.Recorder = recorder

	for attempt := 0; attempt < 2; attempt++ {
		if _, err := reconciler.Reconcile(context.Background(), requestFor(node)); err != nil {
			t.Fatalf("reconcile attempt %d returned an error: %v", attempt+1, err)
		}
	}

	updatedNode := getNode(t, kubernetesClient, node.Name)
	if got := countDegradedTaints(updatedNode); got != 1 {
		t.Fatalf("degraded taint count = %d, want 1", got)
	}
	events := readEvents(t, recorder, 1)
	if !strings.Contains(events[0], "GPUNodeHealthDegraded") {
		t.Fatalf("unexpected event: %q", events[0])
	}
	select {
	case event := <-recorder.Events:
		t.Fatalf("repeated reconcile emitted duplicate event: %q", event)
	default:
	}
}

func TestReconcileContinuesAfterPartialEvictionFailures(t *testing.T) {
	scheme := testScheme(t)
	node := faultedNode("worker-a")
	podA := podOnNode("training", "worker-a", node.Name, remediableLabels("demo-a"))
	podB := podOnNode("training", "worker-b", node.Name, remediableLabels("demo-b"))
	podC := podOnNode("training", "worker-c", node.Name, remediableLabels("demo-c"))
	kubernetesClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(node, podA, podB, podC).Build()

	evictionFailure := errors.New("eviction API unavailable")
	evicter := &recordingEvicter{errors: map[types.NamespacedName]error{
		{Namespace: podA.Namespace, Name: podA.Name}: evictionFailure,
		{Namespace: podB.Namespace, Name: podB.Name}: apierrors.NewNotFound(
			schema.GroupResource{Resource: "pods"},
			podB.Name,
		),
	}}
	reconciler := newTestReconciler(kubernetesClient, evicter)

	_, err := reconciler.Reconcile(context.Background(), requestFor(node))
	if !errors.Is(err, evictionFailure) {
		t.Fatalf("reconcile error = %v, want wrapped eviction failure", err)
	}
	if len(evicter.records) != 3 {
		t.Fatalf("eviction attempts = %#v, want all three pods attempted", evicter.records)
	}
	if !containsEviction(evicter.records, podC.Namespace, podC.Name) {
		t.Fatalf("successful pod after failures was not attempted: %#v", evicter.records)
	}
}

func TestReconcileSkipsTerminatingAndCompletedPods(t *testing.T) {
	scheme := testScheme(t)
	node := faultedNode("worker-a")
	terminating := podOnNode("training", "terminating", node.Name, remediableLabels("demo"))
	now := metav1.NewTime(time.Now())
	terminating.DeletionTimestamp = &now
	terminating.Finalizers = []string{"test.gpu-orch.dev/hold-deletion"}
	succeeded := podOnNode("training", "succeeded", node.Name, remediableLabels("demo"))
	succeeded.Status.Phase = corev1.PodSucceeded
	failed := podOnNode("training", "failed", node.Name, remediableLabels("demo"))
	failed.Status.Phase = corev1.PodFailed
	kubernetesClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		node,
		terminating,
		succeeded,
		failed,
	).Build()
	evicter := &recordingEvicter{}
	reconciler := newTestReconciler(kubernetesClient, evicter)

	if _, err := reconciler.Reconcile(context.Background(), requestFor(node)); err != nil {
		t.Fatalf("reconcile returned an error: %v", err)
	}
	if len(evicter.records) != 0 {
		t.Fatalf("evictions = %#v, want none", evicter.records)
	}
}

func TestReconcileClearsOnlyControllerManagedTaint(t *testing.T) {
	scheme := testScheme(t)
	now := time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC)
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "worker-a",
			Annotations: map[string]string{
				ManagedAnnotation:           "true",
				HealthStateAnnotation:       string(health.StateDegraded),
				RecoveryConfirmedAnnotation: now.Format(time.RFC3339),
			},
		},
		Spec: corev1.NodeSpec{Taints: []corev1.Taint{
			degradedTaint,
			{Key: DegradedTaintKey, Value: "external-value", Effect: corev1.TaintEffectNoSchedule},
			{Key: "example.com/keep", Value: "true", Effect: corev1.TaintEffectNoSchedule},
		}},
	}

	kubernetesClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(node).Build()
	reconciler := newTestReconciler(kubernetesClient, &recordingEvicter{})
	reconciler.Now = func() time.Time { return now }
	reconciler.RecoveryWindow = 20 * time.Second
	if _, err := reconciler.Reconcile(context.Background(), requestFor(node)); err != nil {
		t.Fatalf("first recovery reconcile returned an error: %v", err)
	}
	updatedNode := getNode(t, kubernetesClient, node.Name)
	if !hasDegradedTaint(updatedNode) || updatedNode.Annotations[HealthStateAnnotation] != string(health.StateRecovering) {
		t.Fatalf("first healthy observation did not preserve isolation: %#v", updatedNode)
	}

	now = now.Add(21 * time.Second)
	if _, err := reconciler.Reconcile(context.Background(), requestFor(node)); err != nil {
		t.Fatalf("confirmed recovery reconcile returned an error: %v", err)
	}

	updatedNode = getNode(t, kubernetesClient, node.Name)
	if hasDegradedTaint(updatedNode) {
		t.Fatalf("node retained controller-managed degraded taint: %#v", updatedNode.Spec.Taints)
	}
	if len(updatedNode.Spec.Taints) != 2 {
		t.Fatalf("unrelated taint was changed: %#v", updatedNode.Spec.Taints)
	}
	if _, found := updatedNode.Annotations[ManagedAnnotation]; found {
		t.Fatal("managed annotation was not removed")
	}
}

func TestUnknownObservationNeverClearsManagedIsolation(t *testing.T) {
	scheme := testScheme(t)
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "worker-a",
			Annotations: map[string]string{
				ManagedAnnotation:     "true",
				HealthStateAnnotation: string(health.StateDegraded),
			},
		},
		Spec: corev1.NodeSpec{Taints: []corev1.Taint{degradedTaint}},
	}
	kubernetesClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(node).Build()
	reconciler := newTestReconciler(kubernetesClient, &recordingEvicter{})

	if _, err := reconciler.Reconcile(context.Background(), requestFor(node)); err != nil {
		t.Fatalf("reconcile returned an error: %v", err)
	}
	updatedNode := getNode(t, kubernetesClient, node.Name)
	if !hasDegradedTaint(updatedNode) || updatedNode.Annotations[ManagedAnnotation] != "true" {
		t.Fatalf("unknown observation cleared managed isolation: %#v", updatedNode)
	}
	if updatedNode.Annotations[HealthStateAnnotation] != string(health.StateUnknown) {
		t.Fatalf("health state = %q, want unknown", updatedNode.Annotations[HealthStateAnnotation])
	}
}

func testScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	return scheme
}

func faultedNode(name string) *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Annotations: map[string]string{
				InjectedFaultAnnotation: "xid-79",
			},
		},
	}
}

func newTestReconciler(kubernetesClient client.Client, evicter PodEvicter) *NodeHealthReconciler {
	return &NodeHealthReconciler{
		Client:       kubernetesClient,
		PodReader:    kubernetesClient,
		Evicter:      evicter,
		HealthSource: &healthsource.Annotation{},
	}
}

func podOnNode(namespace, name, nodeName string, labels map[string]string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: name, Labels: labels},
		Spec:       corev1.PodSpec{NodeName: nodeName},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning},
	}
}

func remediableLabels(jobName string) map[string]string {
	return map[string]string{
		TrainingJobLabel:   jobName,
		AutoRemediateLabel: "true",
	}
}

func requestFor(node *corev1.Node) ctrl.Request {
	return ctrl.Request{NamespacedName: types.NamespacedName{Name: node.Name}}
}

func getNode(t *testing.T, kubernetesClient client.Client, name string) *corev1.Node {
	t.Helper()
	node := &corev1.Node{}
	if err := kubernetesClient.Get(
		context.Background(),
		types.NamespacedName{Name: name},
		node,
	); err != nil {
		t.Fatal(err)
	}
	return node
}

func hasDegradedTaint(node *corev1.Node) bool {
	return countDegradedTaints(node) > 0
}

func countDegradedTaints(node *corev1.Node) int {
	count := 0
	for _, taint := range node.Spec.Taints {
		if isDegradedTaint(taint) {
			count++
		}
	}
	return count
}

func containsEviction(records []evictionRecord, namespace, name string) bool {
	for _, record := range records {
		if record.namespace == namespace && record.name == name {
			return true
		}
	}
	return false
}

func readEvents(t *testing.T, recorder *record.FakeRecorder, count int) []string {
	t.Helper()
	events := make([]string, 0, count)
	for len(events) < count {
		select {
		case event := <-recorder.Events:
			events = append(events, event)
		default:
			t.Fatalf("received %d events, want %d", len(events), count)
		}
	}
	return events
}
