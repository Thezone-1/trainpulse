package anomaly

import (
	"testing"
	"time"

	"github.com/somoprovo/trainpulse/internal/model"
)

func TestDetectDataloaderStarvation(t *testing.T) {
	now := time.Now()
	frames := []model.TelemetryFrame{
		{
			Timestamp: now,
			GPUs: []model.GPUSample{{
				Index:       0,
				Utilization: 35,
				MemoryUsed:  1024,
				MemoryTotal: 4096,
				Timestamp:   now,
			}},
			Training: &model.TrainingSample{StepTimeMS: 250, DataWaitMS: 90, Timestamp: now},
		},
	}
	signals := New().Detect(frames)
	if !hasSignal(signals, "dataloader_starvation") {
		t.Fatalf("expected dataloader starvation signal, got %+v", signals)
	}
}

func hasSignal(signals []model.Signal, name string) bool {
	for _, signal := range signals {
		if signal.Name == name {
			return true
		}
	}
	return false
}
