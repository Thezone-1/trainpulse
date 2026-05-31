package anomaly

import (
	"fmt"
	"math"
	"time"

	"github.com/somoprovo/trainpulse/internal/model"
	"github.com/somoprovo/trainpulse/internal/stream"
)

type Engine struct{}

func New() *Engine { return &Engine{} }

func (e *Engine) Detect(frames []model.TelemetryFrame) []model.Signal {
	if len(frames) == 0 {
		return nil
	}
	latest := frames[len(frames)-1]
	now := latest.Timestamp
	var signals []model.Signal
	signals = append(signals, gpuSignals(latest, now)...)
	signals = append(signals, trainingSignals(frames, latest, now)...)
	return signals
}

func gpuSignals(frame model.TelemetryFrame, now time.Time) []model.Signal {
	var signals []model.Signal
	for _, gpu := range frame.GPUs {
		memRatio := ratio(float64(gpu.MemoryUsed), float64(gpu.MemoryTotal))
		if gpu.Utilization < 50 {
			signals = append(signals, model.Signal{
				Name:        "gpu_underutilization",
				Severity:    model.SeverityWarning,
				ScoreImpact: 14,
				Description: fmt.Sprintf("GPU %d utilization is %.1f%%", gpu.Index, gpu.Utilization),
				Evidence:    []string{fmt.Sprintf("gpu=%d util=%.1f%%", gpu.Index, gpu.Utilization)},
				Timestamp:   now,
			})
		}
		if gpu.Temperature >= 84 {
			signals = append(signals, model.Signal{
				Name:        "thermal_instability",
				Severity:    model.SeverityCritical,
				ScoreImpact: 22,
				Description: fmt.Sprintf("GPU %d temperature is %.1fC", gpu.Index, gpu.Temperature),
				Evidence:    []string{fmt.Sprintf("gpu=%d temp=%.1fC clock=%.0fMHz", gpu.Index, gpu.Temperature, gpu.SMClockMHz)},
				Timestamp:   now,
			})
		}
		if memRatio > 0.90 {
			signals = append(signals, model.Signal{
				Name:        "memory_pressure",
				Severity:    model.SeverityWarning,
				ScoreImpact: 16,
				Description: fmt.Sprintf("GPU %d memory is %.1f%% used", gpu.Index, memRatio*100),
				Evidence:    []string{fmt.Sprintf("gpu=%d memory=%d/%dMB", gpu.Index, gpu.MemoryUsed, gpu.MemoryTotal)},
				Timestamp:   now,
			})
		}
	}
	if len(frame.GPUs) >= 2 {
		minUtil, maxUtil := frame.GPUs[0].Utilization, frame.GPUs[0].Utilization
		for _, gpu := range frame.GPUs[1:] {
			minUtil = math.Min(minUtil, gpu.Utilization)
			maxUtil = math.Max(maxUtil, gpu.Utilization)
		}
		if maxUtil-minUtil > 35 {
			signals = append(signals, model.Signal{
				Name:        "synchronization_imbalance",
				Severity:    model.SeverityWarning,
				ScoreImpact: 18,
				Description: fmt.Sprintf("GPU utilization spread is %.1f%%", maxUtil-minUtil),
				Evidence:    []string{fmt.Sprintf("min_util=%.1f%% max_util=%.1f%%", minUtil, maxUtil)},
				Timestamp:   now,
			})
		}
	}
	return signals
}

func trainingSignals(frames []model.TelemetryFrame, latest model.TelemetryFrame, now time.Time) []model.Signal {
	if latest.Training == nil {
		return nil
	}
	tr := latest.Training
	var signals []model.Signal
	avgStep := stream.AvgStepMS(frames)
	if tr.DataWaitMS > 50 && stream.AvgGPUUtil(frames) < 70 {
		signals = append(signals, model.Signal{
			Name:        "dataloader_starvation",
			Severity:    model.SeverityCritical,
			ScoreImpact: 24,
			Description: "High data wait is starving GPU execution",
			Evidence:    []string{fmt.Sprintf("data_wait=%.1fms avg_gpu_util=%.1f%%", tr.DataWaitMS, stream.AvgGPUUtil(frames))},
			Timestamp:   now,
		})
	}
	if tr.SyncWaitMS > 50 {
		signals = append(signals, model.Signal{
			Name:        "communication_bottleneck",
			Severity:    model.SeverityWarning,
			ScoreImpact: 17,
			Description: "Synchronization wait time is elevated",
			Evidence:    []string{fmt.Sprintf("sync_wait=%.1fms", tr.SyncWaitMS)},
			Timestamp:   now,
		})
	}
	if avgStep > 0 && tr.StepTimeMS > avgStep*1.35 {
		signals = append(signals, model.Signal{
			Name:        "throughput_collapse",
			Severity:    model.SeverityWarning,
			ScoreImpact: 15,
			Description: "Current step time is sharply above recent average",
			Evidence:    []string{fmt.Sprintf("step=%.1fms avg_step=%.1fms throughput=%.1f", tr.StepTimeMS, avgStep, tr.Throughput)},
			Timestamp:   now,
		})
	}
	return signals
}

func ratio(n, d float64) float64 {
	if d == 0 {
		return 0
	}
	return n / d
}
