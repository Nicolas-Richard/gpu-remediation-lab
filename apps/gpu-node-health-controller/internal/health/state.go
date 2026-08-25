package health

import "time"

// State is the explicitly observed or decided health of a node's GPUs.
type State string

const (
	StateHealthy    State = "healthy"
	StateDegraded   State = "degraded"
	StateRecovering State = "recovering"
	StateUnknown    State = "unknown"
)

// Observation is one timestamped health fact from one source.
type Observation struct {
	State      State
	Source     string
	Reason     string
	Device     string
	ObservedAt time.Time
}

// ParseState treats missing or unrecognized persisted state as unknown.
func ParseState(value string) State {
	state := State(value)
	switch state {
	case StateHealthy, StateDegraded, StateRecovering, StateUnknown:
		return state
	default:
		return StateUnknown
	}
}
