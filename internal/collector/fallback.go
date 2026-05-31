package collector

import (
	"context"
	"sync/atomic"

	"github.com/somoprovo/trainpulse/internal/model"
)

type Fallback struct {
	primary   Collector
	secondary Collector
	using     atomic.Bool
}

func NewFallback(primary Collector, secondary Collector) *Fallback {
	return &Fallback{primary: primary, secondary: secondary}
}

func (f *Fallback) Name() string {
	if f.using.Load() {
		return f.secondary.Name()
	}
	return f.primary.Name()
}

func (f *Fallback) Collect(ctx context.Context) (model.TelemetryFrame, error) {
	if !f.using.Load() {
		frame, err := f.primary.Collect(ctx)
		if err == nil {
			return frame, nil
		}
		f.using.Store(true)
	}
	return f.secondary.Collect(ctx)
}
