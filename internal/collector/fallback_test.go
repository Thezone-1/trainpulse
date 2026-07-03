package collector

import (
	"context"
	"testing"
	"time"

	"github.com/somoprovo/trainpulse/internal/model"
)

type scriptedCollector struct {
	name  string
	fail  bool
	calls int
}

func (s *scriptedCollector) Name() string { return s.name }

func (s *scriptedCollector) Collect(context.Context) (model.TelemetryFrame, error) {
	s.calls++
	if s.fail {
		return model.TelemetryFrame{}, ErrUnavailable
	}
	return model.TelemetryFrame{Timestamp: time.Now()}, nil
}

func TestFallbackSwitchesAndRetriesPrimary(t *testing.T) {
	primary := &scriptedCollector{name: "nvidia-smi", fail: true}
	secondary := &scriptedCollector{name: "sim"}
	f := NewFallback(primary, secondary)
	ctx := context.Background()

	if _, err := f.Collect(ctx); err != nil {
		t.Fatal(err)
	}
	if f.Name() != "sim" {
		t.Fatalf("expected fallback to secondary, got %s", f.Name())
	}

	// While the retry window is open, the failed primary must not be re-probed
	// on every tick.
	callsAfterSwitch := primary.calls
	if _, err := f.Collect(ctx); err != nil {
		t.Fatal(err)
	}
	if primary.calls != callsAfterSwitch {
		t.Fatal("primary probed again before retry interval elapsed")
	}

	// Once the retry window elapses and the primary recovers, the fallback
	// must return to it instead of serving simulated data forever.
	primary.fail = false
	f.mu.Lock()
	f.nextRetry = time.Now().Add(-time.Second)
	f.mu.Unlock()
	if _, err := f.Collect(ctx); err != nil {
		t.Fatal(err)
	}
	if f.Name() != "nvidia-smi" {
		t.Fatalf("expected recovery to primary, got %s", f.Name())
	}
}
