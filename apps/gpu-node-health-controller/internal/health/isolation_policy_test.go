package health

import (
	"testing"
	"time"
)

func TestDecideIsolationRequiresContinuousConfirmedRecovery(t *testing.T) {
	now := time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC)
	healthy := Observation{State: StateHealthy, ObservedAt: now}

	first := DecideIsolation(StateDegraded, true, nil, healthy, now, 20*time.Second)
	if first.State != StateRecovering || !first.Isolate || first.RecoveryStartedAt == nil {
		t.Fatalf("first healthy decision = %#v, want isolated recovering", first)
	}

	unknown := DecideIsolation(
		first.State,
		first.Isolate,
		first.RecoveryStartedAt,
		Observation{State: StateUnknown, ObservedAt: now.Add(10 * time.Second)},
		now.Add(10*time.Second),
		20*time.Second,
	)
	if unknown.State != StateUnknown || !unknown.Isolate || unknown.RecoveryStartedAt != nil {
		t.Fatalf("unknown decision = %#v, want isolation preserved and recovery reset", unknown)
	}

	restarted := DecideIsolation(
		unknown.State,
		unknown.Isolate,
		unknown.RecoveryStartedAt,
		healthy,
		now.Add(11*time.Second),
		20*time.Second,
	)
	confirmed := DecideIsolation(
		restarted.State,
		restarted.Isolate,
		restarted.RecoveryStartedAt,
		healthy,
		now.Add(32*time.Second),
		20*time.Second,
	)
	if confirmed.State != StateHealthy || confirmed.Isolate {
		t.Fatalf("confirmed decision = %#v, want healthy without isolation", confirmed)
	}
}

func TestDecideIsolationPreservesRecoveryStartAcrossHealthyPolls(t *testing.T) {
	now := time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC)
	healthy := Observation{State: StateHealthy, ObservedAt: now}

	first := DecideIsolation(StateDegraded, true, nil, healthy, now, 20*time.Second)
	second := DecideIsolation(
		first.State,
		first.Isolate,
		first.RecoveryStartedAt,
		healthy,
		now.Add(10*time.Second),
		20*time.Second,
	)
	if second.RecoveryStartedAt == nil || !second.RecoveryStartedAt.Equal(now) {
		t.Fatalf("second healthy poll moved recovery start: %#v", second)
	}

	confirmed := DecideIsolation(
		second.State,
		second.Isolate,
		second.RecoveryStartedAt,
		healthy,
		now.Add(20*time.Second),
		20*time.Second,
	)
	if confirmed.State != StateHealthy || confirmed.Isolate {
		t.Fatalf("confirmed decision = %#v, want healthy without isolation", confirmed)
	}
}

func TestDecideIsolationIsolatesDegradedNodeImmediately(t *testing.T) {
	now := time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC)
	decision := DecideIsolation(
		StateHealthy,
		false,
		nil,
		Observation{State: StateDegraded, ObservedAt: now},
		now,
		20*time.Second,
	)

	if decision.State != StateDegraded || !decision.Isolate {
		t.Fatalf("degraded decision = %#v, want immediate isolation", decision)
	}
}
