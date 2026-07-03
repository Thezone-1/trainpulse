package anomaly

import (
	"testing"
	"time"

	"github.com/somoprovo/trainpulse/internal/config"
	"github.com/somoprovo/trainpulse/internal/model"
)

func TestRuleDetector(t *testing.T) {
	now := time.Now()
	detector := NewRuleDetector([]config.Rule{
		{
			Name:        "team_low_mfu",
			Field:       "training.mfu",
			Operator:    "lt",
			Value:       0.40,
			Severity:    "warning",
			ScoreImpact: 11,
			Description: "MFU below team target",
		},
		{
			Name:        "gpu_hot",
			Field:       "gpu.temperature_c",
			Operator:    "gt",
			Value:       80,
			Severity:    "critical",
			ScoreImpact: 15,
		},
	})
	signals := detector.Detect([]model.TelemetryFrame{{
		Timestamp: now,
		GPUs: []model.GPUSample{
			{Index: 0, Temperature: 82, Timestamp: now},
		},
		Training: &model.TrainingSample{
			MFU:       0.31,
			Timestamp: now,
		},
	}})
	for _, name := range []string{"team_low_mfu", "gpu_hot"} {
		if !hasSignal(signals, name) {
			t.Fatalf("expected %s, got %+v", name, signals)
		}
	}
}
