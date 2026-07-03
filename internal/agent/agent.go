package agent

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/somoprovo/trainpulse/internal/anomaly"
	"github.com/somoprovo/trainpulse/internal/collector"
	"github.com/somoprovo/trainpulse/internal/config"
	"github.com/somoprovo/trainpulse/internal/correlate"
	"github.com/somoprovo/trainpulse/internal/diagnostics"
	"github.com/somoprovo/trainpulse/internal/health"
	"github.com/somoprovo/trainpulse/internal/knowledge"
	"github.com/somoprovo/trainpulse/internal/model"
	"github.com/somoprovo/trainpulse/internal/stream"
)

// trainingStaleAfter bounds how long a pushed training sample keeps being
// attached to new telemetry frames after the training loop stops reporting.
const trainingStaleAfter = 30 * time.Second

type Agent struct {
	cfg        config.Config
	collector  collector.Collector
	logger     *slog.Logger
	window     *stream.Window
	anomaly    *anomaly.Engine
	correlator *correlate.Correlator
	scorer     *health.Scorer
	inferer    *diagnostics.Inferer

	mu            sync.RWMutex
	snapshot      model.Snapshot
	count         int64
	collectErrors int64
	lastError     string
	lastLogged    string

	trainingMu       sync.RWMutex
	latestTraining   *model.TrainingSample
	trainingReceived time.Time
}

func New(cfg config.Config, col collector.Collector) *Agent {
	anomalyEngine := anomaly.New()
	if len(cfg.Rules) > 0 {
		anomalyEngine.Register(anomaly.NewRuleDetector(cfg.Rules))
	}
	return &Agent{
		cfg:        cfg,
		collector:  col,
		logger:     slog.Default(),
		window:     stream.NewWindow(cfg.HistorySize),
		anomaly:    anomalyEngine,
		correlator: correlate.New(),
		scorer:     health.New(),
		inferer:    diagnostics.NewWithBase(knowledge.New(cfg.Diagnoses)),
	}
}

// SetLogger overrides the logger used for collection failures.
func (a *Agent) SetLogger(l *slog.Logger) {
	if l != nil {
		a.logger = l
	}
}

// Run collects telemetry until the context is cancelled. A failing collector
// does not stop the loop: the daemon keeps serving the last snapshot and
// records the failure, because a monitoring agent that dies on a transient
// nvidia-smi hiccup is worse than one reporting a degraded collector.
func (a *Agent) Run(ctx context.Context) error {
	ticker := time.NewTicker(a.cfg.Interval)
	defer ticker.Stop()
	a.tickLogged(ctx)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			a.tickLogged(ctx)
		}
	}
}

// tickLogged logs a collection failure at warn only when the error changes;
// identical repeats drop to debug so a dead collector cannot flood the log at
// every interval.
func (a *Agent) tickLogged(ctx context.Context) {
	err := a.Tick(ctx)
	if err == nil {
		a.mu.Lock()
		if a.lastLogged != "" {
			a.lastLogged = ""
			a.mu.Unlock()
			a.logger.Info("collect_recovered", "collector", a.collector.Name())
			return
		}
		a.mu.Unlock()
		return
	}
	if ctx.Err() != nil {
		return
	}
	a.mu.Lock()
	repeat := a.lastLogged == err.Error()
	a.lastLogged = err.Error()
	a.mu.Unlock()
	if repeat {
		a.logger.Debug("collect_failed", "error", err, "collector", a.collector.Name())
		return
	}
	a.logger.Warn("collect_failed", "error", err, "collector", a.collector.Name())
}

func (a *Agent) Snapshot() model.Snapshot {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.snapshot
}

func (a *Agent) UpdateTraining(sample model.TrainingSample) {
	now := time.Now()
	if sample.Timestamp.IsZero() {
		sample.Timestamp = now
	}
	a.trainingMu.Lock()
	a.latestTraining = &sample
	a.trainingReceived = now
	a.trainingMu.Unlock()
}

func (a *Agent) Tick(ctx context.Context) error {
	frame, err := a.collector.Collect(ctx)
	if err != nil {
		a.mu.Lock()
		a.collectErrors++
		a.lastError = err.Error()
		a.snapshot.CollectErrors = a.collectErrors
		a.snapshot.LastError = a.lastError
		a.mu.Unlock()
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
	collectorName := a.collector.Name()
	a.mu.Lock()
	a.count++
	a.snapshot = model.Snapshot{
		Timestamp:     frame.Timestamp,
		Health:        score,
		Status:        status,
		Telemetry:     frame,
		Signals:       signals,
		Diagnoses:     diagnoses,
		SampleCount:   a.count,
		Collector:     collectorName,
		Simulated:     collectorName == "sim",
		CollectErrors: a.collectErrors,
		LastError:     a.lastError,
	}
	a.mu.Unlock()
	return nil
}

// recentTraining returns the last pushed training sample, judged fresh by the
// daemon's own receive clock so client clock skew cannot drop or pin samples.
func (a *Agent) recentTraining() *model.TrainingSample {
	a.trainingMu.RLock()
	defer a.trainingMu.RUnlock()
	if a.latestTraining == nil || time.Since(a.trainingReceived) > trainingStaleAfter {
		return nil
	}
	sample := *a.latestTraining
	return &sample
}
