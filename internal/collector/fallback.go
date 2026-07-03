package collector

import (
	"context"
	"sync"
	"time"

	"github.com/somoprovo/trainpulse/internal/model"
)

// primaryRetryInterval is how often the fallback re-probes a failed primary.
// Without periodic retries a single transient nvidia-smi failure would pin a
// real GPU host to the secondary (often the simulator) for the daemon's whole
// lifetime, silently serving fake telemetry.
const primaryRetryInterval = 30 * time.Second

type Fallback struct {
	primary   Collector
	secondary Collector

	mu        sync.Mutex
	usingSec  bool
	nextRetry time.Time
}

func NewFallback(primary Collector, secondary Collector) *Fallback {
	return &Fallback{primary: primary, secondary: secondary}
}

func (f *Fallback) Name() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.usingSec {
		return f.secondary.Name()
	}
	return f.primary.Name()
}

func (f *Fallback) Collect(ctx context.Context) (model.TelemetryFrame, error) {
	f.mu.Lock()
	tryPrimary := !f.usingSec || time.Now().After(f.nextRetry)
	f.mu.Unlock()

	if tryPrimary {
		frame, err := f.primary.Collect(ctx)
		if err == nil {
			f.mu.Lock()
			f.usingSec = false
			f.mu.Unlock()
			return frame, nil
		}
		f.mu.Lock()
		f.usingSec = true
		f.nextRetry = time.Now().Add(primaryRetryInterval)
		f.mu.Unlock()
	}
	return f.secondary.Collect(ctx)
}
