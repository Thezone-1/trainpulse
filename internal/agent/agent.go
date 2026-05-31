package agent

import (
	"context"
	"sync"
	"time"

	"github.com/somoprovo/trainpulse/internal/anomaly"
	"github.com/somoprovo/trainpulse/internal/collector"
	"github.com/somoprovo/trainpulse/internal/config"
	"github.com/somoprovo/trainpulse/internal/correlate"
	"github.com/somoprovo/trainpulse/internal/diagnostics"
	"github.com/somoprovo/trainpulse/internal/health"
	"github.com/somoprovo/trainpulse/internal/model"
	"github.com/somoprovo/trainpulse/internal/stream"
)

type Agent struct {
	cfg        config.Config
	collector  collector.Collector
	window     *stream.Window
	anomaly    *anomaly.Engine
	correlator *correlate.Correlator
	scorer     *health.Scorer
	inferer    *diagnostics.Inferer

	mu       sync.RWMutex
	snapshot model.Snapshot
	count    int64

	trainingMu     sync.RWMutex
	latestTraining *model.TrainingSample
}

func New(cfg config.Config, col collector.Collector) *Agent {
	return &Agent{
		cfg:        cfg,
		collector:  col,
		window:     stream.NewWindow(cfg.HistorySize),
		anomaly:    anomaly.New(),
		correlator: correlate.New(),
		scorer:     health.New(),
		inferer:    diagnostics.New(),
	}
}

func (a *Agent) Run(ctx context.Context) error {
	ticker := time.NewTicker(a.cfg.Interval)
	defer ticker.Stop()
	if err := a.Tick(ctx); err != nil {
		return err
	}
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := a.Tick(ctx); err != nil {
				return err
			}
		}
	}
}

func (a *Agent) Snapshot() model.Snapshot {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.snapshot
}

func (a *Agent) UpdateTraining(sample model.TrainingSample) {
	if sample.Timestamp.IsZero() {
		sample.Timestamp = time.Now()
	}
	a.trainingMu.Lock()
	a.latestTraining = &sample
	a.trainingMu.Unlock()
}

func (a *Agent) Tick(ctx context.Context) error {
	frame, err := a.collector.Collect(ctx)
	if err != nil {
		return err
	}
	if frame.Training == nil {
		frame.Training = a.recentTraining()
	}
	a.window.Add(frame)
	frames := a.window.Frames()
	signals := a.correlator.Correlate(a.anomaly.Detect(frames))
	score, status := a.scorer.Score(signals)
	diagnoses := a.inferer.Infer(signals)
	a.count++
	snap := model.Snapshot{
		Timestamp:   frame.Timestamp,
		Health:      score,
		Status:      status,
		Telemetry:   frame,
		Signals:     signals,
		Diagnoses:   diagnoses,
		SampleCount: a.count,
	}
	a.mu.Lock()
	a.snapshot = snap
	a.mu.Unlock()
	return nil
}

func (a *Agent) recentTraining() *model.TrainingSample {
	a.trainingMu.RLock()
	defer a.trainingMu.RUnlock()
	if a.latestTraining == nil || time.Since(a.latestTraining.Timestamp) > 30*time.Second {
		return nil
	}
	sample := *a.latestTraining
	return &sample
}
