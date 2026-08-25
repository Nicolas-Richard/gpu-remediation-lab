package health

import "time"

// decideRecovery preserves isolation until healthy evidence has remained
// continuous for the configured confirmation window.
func decideRecovery(
	previous State,
	isolated bool,
	recoveryStartedAt *time.Time,
	now time.Time,
	recoveryWindow time.Duration,
) IsolationDecision {
	if !isolated {
		return IsolationDecision{State: StateHealthy}
	}
	if previous == StateRecovering && recoveryStartedAt != nil {
		started := recoveryStartedAt.UTC()
		if !now.Before(started.Add(recoveryWindow)) {
			return IsolationDecision{State: StateHealthy}
		}
		return IsolationDecision{
			State:             StateRecovering,
			Isolate:           true,
			RecoveryStartedAt: &started,
		}
	}
	return continueRecovery(nil, now)
}

func continueRecovery(recoveryStartedAt *time.Time, now time.Time) IsolationDecision {
	started := now
	if recoveryStartedAt != nil {
		started = recoveryStartedAt.UTC()
	}
	return IsolationDecision{
		State:             StateRecovering,
		Isolate:           true,
		RecoveryStartedAt: &started,
	}
}
