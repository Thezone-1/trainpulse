package agent

import (
	"context"
	"testing"
	"time"

	"github.com/somoprovo/trainpulse/internal/config"
	"github.com/somoprovo/trainpulse/internal/model"
)

func TestTickRunsDiagnosticsPipeline(t *testing.T) {
	now := time.Now()
	a := New(config.Config{Interval: time.Second, HistorySize: 8}, staticCollector{
		frame: model.TelemetryFrame{
			Timestamp: now,
			GPUs: []model.GPUSample{
				{Index: 0, Utilization: 35, MemoryUsed: 1900, MemoryTotal: 2000, Temperature: 86, Timestamp: now},
				{Index: 1, Utilization: 92, MemoryUsed: 1000, MemoryTotal: 2000, Temperature: 68, Timestamp: now},
			},
			Training: &model.TrainingSample{
				StepTimeMS:      260,
				TokensPerSec:    30000,
				MFU:             0.22,
				DataWaitMS:      90,
				TokenizerWaitMS: 75,
				AllReduceWaitMS: 80,
				AvgSeqLen:       600,
				MaxSeqLen:       2048,
				WorldSize:       8,
				Timestamp:       now,
			},
		},
	})
	if err := a.Tick(context.Background()); err != nil {
		t.Fatal(err)
	}
	snap := a.Snapshot()
	if snap.SampleCount != 1 {
		t.Fatalf("expected one sample, got %d", snap.SampleCount)
	}
	if snap.Health >= 100 {
		t.Fatalf("expected degraded health, got %.1f", snap.Health)
	}
	for _, name := range []string{"thermal_instability", "dataloader_starvation", "low_mfu", "tokenizer_bottleneck", "allreduce_bottleneck"} {
		if !hasSignal(snap.Signals, name) {
			t.Fatalf("expected signal %s, got %+v", name, snap.Signals)
		}
	}
	if len(snap.Diagnoses) == 0 {
		t.Fatal("expected root-cause diagnoses")
	}
}

func TestTickMergesFreshRuntimeTrainingSample(t *testing.T) {
	now := time.Now()
	a := New(config.Config{Interval: time.Second, HistorySize: 8}, staticCollector{
		frame: model.TelemetryFrame{
			Timestamp: now,
			GPUs:      []model.GPUSample{{Index: 0, Utilization: 80, MemoryUsed: 1000, MemoryTotal: 2000, Timestamp: now}},
		},
	})
	a.UpdateTraining(model.TrainingSample{ModelName: "llama-runtime", TokensPerSec: 42000})
	if err := a.Tick(context.Background()); err != nil {
		t.Fatal(err)
	}
	tr := a.Snapshot().Telemetry.Training
	if tr == nil || tr.ModelName != "llama-runtime" {
		t.Fatalf("expected merged runtime training sample, got %+v", tr)
	}
}

func TestTickRunsConfiguredRules(t *testing.T) {
	now := time.Now()
	a := New(config.Config{
		Interval:    time.Second,
		HistorySize: 8,
		Rules: []config.Rule{
			{Name: "team_tokens_slo", Field: "training.tokens_per_sec", Operator: "lt", Value: 50000, Severity: "warning", ScoreImpact: 9},
		},
	}, staticCollector{
		frame: model.TelemetryFrame{
			Timestamp: now,
			GPUs:      []model.GPUSample{{Index: 0, Utilization: 80, MemoryUsed: 1000, MemoryTotal: 2000, Timestamp: now}},
			Training:  &model.TrainingSample{TokensPerSec: 42000, Timestamp: now},
		},
	})
	if err := a.Tick(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !hasSignal(a.Snapshot().Signals, "team_tokens_slo") {
		t.Fatalf("expected configured rule signal, got %+v", a.Snapshot().Signals)
	}
}

func TestTickRecordsCollectorFailureWithoutLosingSnapshot(t *testing.T) {
	now := time.Now()
	col := &flakyCollector{
		frame: model.TelemetryFrame{
			Timestamp: now,
			GPUs:      []model.GPUSample{{Index: 0, Utilization: 80, MemoryUsed: 1000, MemoryTotal: 2000, Timestamp: now}},
		},
	}
	a := New(config.Config{Interval: time.Second, HistorySize: 8}, col)
	if err := a.Tick(context.Background()); err != nil {
		t.Fatal(err)
	}
	col.fail = true
	if err := a.Tick(context.Background()); err == nil {
		t.Fatal("expected collect error")
	}
	snap := a.Snapshot()
	if snap.SampleCount != 1 {
		t.Fatalf("failed tick must not drop the last good snapshot, got count %d", snap.SampleCount)
	}
	if snap.CollectErrors != 1 || snap.LastError == "" {
		t.Fatalf("expected recorded collect failure, got %+v", snap)
	}
	if len(snap.Telemetry.GPUs) != 1 {
		t.Fatal("last good telemetry should still be served")
	}
}

func TestSnapshotMarksSimulatedCollector(t *testing.T) {
	a := New(config.Config{Interval: time.Second, HistorySize: 8}, namedCollector{name: "sim"})
	if err := a.Tick(context.Background()); err != nil {
		t.Fatal(err)
	}
	snap := a.Snapshot()
	if snap.Collector != "sim" || !snap.Simulated {
		t.Fatalf("expected simulated provenance, got collector=%q simulated=%v", snap.Collector, snap.Simulated)
	}
}

type flakyCollector struct {
	frame model.TelemetryFrame
	fail  bool
}

func (f *flakyCollector) Name() string { return "flaky" }

func (f *flakyCollector) Collect(context.Context) (model.TelemetryFrame, error) {
	if f.fail {
		return model.TelemetryFrame{}, context.DeadlineExceeded
	}
	return f.frame, nil
}

type namedCollector struct {
	name string
}

func (n namedCollector) Name() string { return n.name }

func (n namedCollector) Collect(context.Context) (model.TelemetryFrame, error) {
	return model.TelemetryFrame{Timestamp: time.Now()}, nil
}

type staticCollector struct {
	frame model.TelemetryFrame
}

func (s staticCollector) Name() string { return "static" }

func (s staticCollector) Collect(context.Context) (model.TelemetryFrame, error) {
	return s.frame, nil
}

func hasSignal(signals []model.Signal, name string) bool {
	for _, signal := range signals {
		if signal.Name == name {
			return true
		}
	}
	return false
}
