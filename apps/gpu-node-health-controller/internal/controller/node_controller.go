package controller

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	"hackweek/gpu-node-health-controller/internal/health"
	healthsource "hackweek/gpu-node-health-controller/internal/health/source"
)

const (
	// InjectedFaultAnnotation is the synthetic health signal watched by this
	// controller. A non-empty value means the node should be remediated.
	InjectedFaultAnnotation = healthsource.SyntheticFaultAnnotation
	// InjectedFaultDeviceAnnotation optionally identifies the synthetic GPU that
	// produced the fault. The Phase 2 harness leaves it unset, so events report
	// the device as unknown while preserving a stable contract for Phase 3.
	InjectedFaultDeviceAnnotation = healthsource.SyntheticDeviceAnnotation
	// RecoveryConfirmedAnnotation is an explicit healthy observation. Merely
	// removing a fault signal is not sufficient to clear isolation.
	RecoveryConfirmedAnnotation = healthsource.SyntheticRecoveryAnnotation
	// Health annotations expose the state machine and its evidence on each Node.
	HealthStateAnnotation       = "gpu-orch.dev/health-state"
	HealthSourceAnnotation      = "gpu-orch.dev/health-source"
	HealthReasonAnnotation      = "gpu-orch.dev/health-reason"
	HealthDeviceAnnotation      = "gpu-orch.dev/health-device"
	HealthObservedAtAnnotation  = "gpu-orch.dev/health-observed-at"
	RecoveryStartedAtAnnotation = "gpu-orch.dev/recovery-started-at"
)

// NodeHealthReconciler converts a health observation into a scheduling taint and
// eviction requests for opted-in training pods currently running on that node.
type NodeHealthReconciler struct {
	// Client reads and updates nodes through controller-runtime's cached client.
	client.Client
	// PodReader reads current pod placement directly from the Kubernetes API.
	PodReader client.Reader
	// Evicter submits policy/v1 Eviction requests.
	Evicter PodEvicter
	// Recorder publishes human-readable Kubernetes Events for remediation actions.
	Recorder record.EventRecorder
	// HealthSource is the configured provider of node health observations.
	HealthSource healthsource.Source
	// PollInterval bounds how long metric-only state changes take to reconcile.
	PollInterval time.Duration
	// RecoveryWindow is the continuous healthy period required before untainting.
	RecoveryWindow time.Duration
	// Now is replaceable for deterministic state-machine tests.
	Now func() time.Time
}

// Reconcile makes node isolation match one observation from the shared health
// decision pipeline. Periodic requeues are required because metric changes do
// not produce Kubernetes Node events.
func (r *NodeHealthReconciler) Reconcile(ctx context.Context, request ctrl.Request) (ctrl.Result, error) {
	logger := ctrl.LoggerFrom(ctx).WithValues("node", request.Name)
	node := &corev1.Node{}
	// The request contains a node name, but not the Node object itself.
	if err := r.Get(ctx, types.NamespacedName{Name: request.Name}, node); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	now := time.Now
	if r.Now != nil {
		now = r.Now
	}
	reconciledAt := now().UTC()
	if r.HealthSource == nil {
		return ctrl.Result{}, errors.New("health source is not configured")
	}
	observation := r.HealthSource.Observe(ctx, node)
	if observation.ObservedAt.IsZero() {
		observation.ObservedAt = reconciledAt
	}
	previousState := health.ParseState(node.Annotations[HealthStateAnnotation])
	managed := node.Annotations[ManagedAnnotation] == "true"
	recoveryStartedAt := annotationTime(node, RecoveryStartedAtAnnotation)
	recoveryWindow := r.RecoveryWindow
	if recoveryWindow <= 0 {
		recoveryWindow = 20 * time.Second
	}
	decision := health.DecideIsolation(
		previousState,
		managed,
		recoveryStartedAt,
		observation,
		reconciledAt,
		recoveryWindow,
	)

	changed := applyHealthAnnotations(node, decision, observation)
	var taintAdded, taintRemoved bool
	if decision.Isolate {
		isolation := isolateNode(node)
		changed = changed || isolation.Changed
		taintAdded = isolation.TaintAdded
	} else {
		recovery := recoverNode(node)
		changed = changed || recovery.Changed
		taintRemoved = recovery.TaintRemoved
	}
	if changed {
		if err := r.Update(ctx, node); err != nil {
			return ctrl.Result{}, err
		}
	}

	device := strings.TrimSpace(observation.Device)
	if device == "" {
		device = "unknown"
	}
	if taintAdded {
		r.eventf(
			node,
			corev1.EventTypeWarning,
			"GPUNodeHealthDegraded",
			"node=%s device=%s source=%s state=%s observedAt=%s reason=%s action=add-taint:%s",
			node.Name,
			device,
			observation.Source,
			decision.State,
			observation.ObservedAt.Format(time.RFC3339Nano),
			observation.Reason,
			DegradedTaintKey,
		)
		logger.Info("marked node degraded", "reason", observation.Reason, "source", observation.Source)
	}
	if decision.State == health.StateRecovering && previousState != health.StateRecovering {
		r.eventf(
			node,
			corev1.EventTypeNormal,
			"GPUNodeHealthRecovering",
			"node=%s device=%s source=%s state=%s observedAt=%s reason=%s action=preserve-isolation",
			node.Name,
			device,
			observation.Source,
			decision.State,
			observation.ObservedAt.Format(time.RFC3339Nano),
			observation.Reason,
		)
	}
	if decision.State == health.StateUnknown && previousState != health.StateUnknown {
		r.eventf(
			node,
			corev1.EventTypeWarning,
			"GPUNodeHealthUnknown",
			"node=%s device=%s source=%s state=%s observedAt=%s reason=%s action=preserve-isolation:%t",
			node.Name,
			device,
			observation.Source,
			decision.State,
			observation.ObservedAt.Format(time.RFC3339Nano),
			observation.Reason,
			decision.Isolate,
		)
	}
	if taintRemoved {
		r.eventf(
			node,
			corev1.EventTypeNormal,
			"GPUNodeHealthRecovered",
			"node=%s device=%s source=%s state=%s observedAt=%s reason=%s action=remove-taint:%s",
			node.Name,
			device,
			observation.Source,
			decision.State,
			observation.ObservedAt.Format(time.RFC3339Nano),
			observation.Reason,
			DegradedTaintKey,
		)
		logger.Info("confirmed node health recovery", "source", observation.Source)
	}

	pollResult := ctrl.Result{RequeueAfter: r.pollInterval()}
	if decision.State != health.StateDegraded {
		return pollResult, nil
	}

	// Taint before evicting so Kubernetes cannot place a replacement pod back on
	// the degraded node during the eviction/recreation window.
	// Pods are namespaced resources, so list across all namespaces and then keep
	// only active Kubeflow training pods assigned to this node.
	pods := &corev1.PodList{}
	if err := r.PodReader.List(ctx, pods); err != nil {
		return ctrl.Result{}, err
	}
	var evictionErrors []error
	for i := range pods.Items {
		pod := &pods.Items[i]
		if !isImpactedTrainingPod(pod, node.Name) {
			continue
		}
		// NotFound is already the desired outcome and makes retries idempotent.
		if err := r.Evicter.Evict(ctx, pod.Namespace, pod.Name); err != nil {
			if apierrors.IsNotFound(err) {
				continue
			}
			r.eventf(
				node,
				corev1.EventTypeWarning,
				"TrainingPodEvictionFailed",
				"node=%s device=%s source=%s state=%s observedAt=%s reason=%s action=evict-failed pod=%s/%s error=%s",
				node.Name,
				device,
				observation.Source,
				decision.State,
				observation.ObservedAt.Format(time.RFC3339Nano),
				observation.Reason,
				pod.Namespace,
				pod.Name,
				err,
			)
			evictionErrors = append(
				evictionErrors,
				fmt.Errorf("evict training pod %s/%s: %w", pod.Namespace, pod.Name, err),
			)
			continue
		}
		r.eventf(
			node,
			corev1.EventTypeNormal,
			"TrainingPodEvictionRequested",
			"node=%s device=%s source=%s state=%s observedAt=%s reason=%s action=evict pod=%s/%s",
			node.Name,
			device,
			observation.Source,
			decision.State,
			observation.ObservedAt.Format(time.RFC3339Nano),
			observation.Reason,
			pod.Namespace,
			pod.Name,
		)
		logger.Info("requested training pod eviction", "namespace", pod.Namespace, "pod", pod.Name)
	}

	return pollResult, errors.Join(evictionErrors...)
}

// SetupWithManager registers this reconciler to receive Node events.
func (r *NodeHealthReconciler) SetupWithManager(manager ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(manager).
		For(&corev1.Node{}).
		WithEventFilter(predicate.Funcs{
			CreateFunc: func(event.CreateEvent) bool { return true },
			UpdateFunc: func(update event.UpdateEvent) bool {
				oldNode, oldOK := update.ObjectOld.(*corev1.Node)
				newNode, newOK := update.ObjectNew.(*corev1.Node)
				if !oldOK || !newOK {
					return false
				}
				for _, annotation := range []string{
					InjectedFaultAnnotation,
					InjectedFaultDeviceAnnotation,
					RecoveryConfirmedAnnotation,
				} {
					if oldNode.Annotations[annotation] != newNode.Annotations[annotation] {
						return true
					}
				}
				return false
			},
			DeleteFunc:  func(event.DeleteEvent) bool { return false },
			GenericFunc: func(event.GenericEvent) bool { return false },
		}).
		Complete(r)
}

func (r *NodeHealthReconciler) pollInterval() time.Duration {
	if r.PollInterval > 0 {
		return r.PollInterval
	}
	return 10 * time.Second
}

func applyHealthAnnotations(
	node *corev1.Node,
	decision health.IsolationDecision,
	observation health.Observation,
) bool {
	if node.Annotations == nil {
		node.Annotations = make(map[string]string)
	}
	changed := false
	changed = setAnnotation(node, HealthStateAnnotation, string(decision.State)) || changed
	changed = setAnnotation(node, HealthSourceAnnotation, observation.Source) || changed
	changed = setAnnotation(node, HealthReasonAnnotation, observation.Reason) || changed
	changed = setAnnotation(node, HealthDeviceAnnotation, observation.Device) || changed
	changed = setAnnotation(
		node,
		HealthObservedAtAnnotation,
		observation.ObservedAt.UTC().Format(time.RFC3339Nano),
	) || changed
	if decision.RecoveryStartedAt != nil {
		changed = setAnnotation(
			node,
			RecoveryStartedAtAnnotation,
			decision.RecoveryStartedAt.UTC().Format(time.RFC3339Nano),
		) || changed
	} else if _, found := node.Annotations[RecoveryStartedAtAnnotation]; found {
		delete(node.Annotations, RecoveryStartedAtAnnotation)
		changed = true
	}
	return changed
}

func setAnnotation(node *corev1.Node, key, value string) bool {
	if node.Annotations[key] == value {
		return false
	}
	node.Annotations[key] = value
	return true
}

func annotationTime(node *corev1.Node, key string) *time.Time {
	value := strings.TrimSpace(node.Annotations[key])
	if value == "" {
		return nil
	}
	parsed, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		return nil
	}
	return &parsed
}

func (r *NodeHealthReconciler) eventf(
	node *corev1.Node,
	eventType string,
	reason string,
	message string,
	args ...interface{},
) {
	if r.Recorder != nil {
		r.Recorder.Eventf(node, eventType, reason, message, args...)
	}
}
