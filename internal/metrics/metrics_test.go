package metrics

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/somoprovo/trainpulse/internal/model"
)

func multiGPUSnapshot() model.Snapshot {
	now := time.Now()
	return model.Snapshot{
		Timestamp:   now,
		Health:      72,
		Status:      model.SeverityWarning,
		SampleCount: 42,
		Telemetry: model.TelemetryFrame{
			Timestamp: now,
			GPUs: []model.GPUSample{
				{Index: 0, Name: "H100", Utilization: 91, MemoryUsed: 17000, MemoryTotal: 24576, Temperature: 66, PowerWatts: 410},
				{Index: 1, Name: "H100", Utilization: 88, MemoryUsed: 17200, MemoryTotal: 24576, Temperature: 64, PowerWatts: 400},
			},
			Training: &model.TrainingSample{
				ModelName: "llama-7b", Framework: "pytorch", StepTimeMS: 180, TokensPerSec: 72000, MFU: 0.42,
			},
		},
		Signals: []model.Signal{
			{Name: "low_mfu", Severity: model.SeverityWarning},
			{Name: "gpu_underutilization", Severity: model.SeverityWarning},
		},
	}
}

func TestWritePrometheusGroupsFamilies(t *testing.T) {
	var buf bytes.Buffer
	if err := WritePrometheus(&buf, multiGPUSnapshot(), "trainpulse"); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	// The text format allows # HELP and # TYPE at most once per metric family.
	seenType := map[string]bool{}
	for _, line := range strings.Split(out, "\n") {
		if !strings.HasPrefix(line, "# TYPE ") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) != 4 {
			t.Fatalf("malformed TYPE line: %q", line)
		}
		name := fields[2]
		if seenType[name] {
			t.Fatalf("duplicate # TYPE for %s:\n%s", name, out)
		}
		seenType[name] = true
	}
	if !seenType["trainpulse_gpu_utilization_percent"] {
		t.Fatal("missing gpu utilization family")
	}
	if got := strings.Count(out, "trainpulse_gpu_utilization_percent{"); got != 2 {
		t.Fatalf("expected 2 gpu utilization samples, got %d:\n%s", got, out)
	}
	if got := strings.Count(out, "trainpulse_signal_active{"); got != 2 {
		t.Fatalf("expected 2 signal samples, got %d", got)
	}
	if !strings.Contains(out, "# TYPE trainpulse_samples_total counter") {
		t.Fatalf("samples_total should be a counter:\n%s", out)
	}
}

func TestWritePrometheusDeterministic(t *testing.T) {
	snap := multiGPUSnapshot()
	var first bytes.Buffer
	if err := WritePrometheus(&first, snap, "trainpulse"); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		var again bytes.Buffer
		if err := WritePrometheus(&again, snap, "trainpulse"); err != nil {
			t.Fatal(err)
		}
		if again.String() != first.String() {
			t.Fatal("exposition output is not deterministic across renders")
		}
	}
}

func TestEscapeLabel(t *testing.T) {
	got := escapeLabel("a\"b\\c\nd")
	want := `a\"b\\c\nd`
	if got != want {
		t.Fatalf("escapeLabel = %q, want %q", got, want)
	}
}
