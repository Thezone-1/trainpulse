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

func TestDetectLLMRanksAndPadding(t *testing.T) {
	now := time.Now()
	frames := []model.TelemetryFrame{
		{
			Timestamp: now,
			Training: &model.TrainingSample{
				WorkloadKind: "llm_pretraining",
				StepTimeMS:   180,
				TokensPerSec: 42000,
				MFU:          0.24,
				AvgSeqLen:    700,
				MaxSeqLen:    2048,
				Ranks: []model.RankSample{
					{Rank: 0, StepTimeMS: 150},
					{Rank: 1, StepTimeMS: 225},
				},
				Timestamp: now,
			},
		},
	}
	signals := New().Detect(frames)
	for _, name := range []string{"low_mfu", "excessive_padding", "rank_straggler"} {
		if !hasSignal(signals, name) {
			t.Fatalf("expected %s signal, got %+v", name, signals)
		}
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
