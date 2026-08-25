package health

import "time"

// IsolationDecision says whether the node must remain isolated after applying
// the current observation to its previous health state.
type IsolationDecision struct {
	State             State
	Isolate           bool
	RecoveryStartedAt *time.Time
}

// DecideIsolation applies the same isolation policy regardless of observation
// source. Degraded evidence isolates immediately, unknown evidence never clears
// existing isolation, and healthy evidence delegates to the recovery policy.
func DecideIsolation(
	previous State,
	isolated bool,
	recoveryStartedAt *time.Time,
	observation Observation,
	now time.Time,
	recoveryWindow time.Duration,
) IsolationDecision {
	now = now.UTC()
	switch observation.State {
	case StateDegraded:
		return IsolationDecision{State: StateDegraded, Isolate: true}
	case StateUnknown:
		return IsolationDecision{State: StateUnknown, Isolate: isolated}
	case StateRecovering:
		return continueRecovery(recoveryStartedAt, now)
	case StateHealthy:
		return decideRecovery(previous, isolated, recoveryStartedAt, now, recoveryWindow)
	default:
		return IsolationDecision{State: StateUnknown, Isolate: isolated}
	}
}
