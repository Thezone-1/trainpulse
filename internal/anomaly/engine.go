package anomaly

import (
	"fmt"
	"math"
	"time"

	"github.com/somoprovo/trainpulse/internal/model"
	"github.com/somoprovo/trainpulse/internal/stream"
)

type Detector interface {
	Name() string
	Detect([]model.TelemetryFrame) []model.Signal
}

type DetectorFunc struct {
	NameValue string
	Fn        func([]model.TelemetryFrame) []model.Signal
}

func (d DetectorFunc) Name() string { return d.NameValue }

func (d DetectorFunc) Detect(frames []model.TelemetryFrame) []model.Signal {
	if d.Fn == nil {
		return nil
	}
	return d.Fn(frames)
}

type Engine struct {
	detectors []Detector
}

func New(detectors ...Detector) *Engine { return &Engine{detectors: detectors} }

func (e *Engine) Register(detector Detector) {
	e.detectors = append(e.detectors, detector)
}

func (e *Engine) Detect(frames []model.TelemetryFrame) []model.Signal {
	if len(frames) == 0 {
		return nil
	}
	latest := frames[len(frames)-1]
	now := latest.Timestamp
	var signals []model.Signal
	signals = append(signals, gpuSignals(latest, now)...)
	signals = append(signals, trainingSignals(frames, latest, now)...)
	for _, detector := range e.detectors {
		signals = append(signals, detector.Detect(frames)...)
	}
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
	if tr.TokenizerWaitMS > 50 {
		signals = append(signals, model.Signal{
			Name:        "tokenizer_bottleneck",
			Severity:    model.SeverityWarning,
			ScoreImpact: 14,
			Description: "Tokenizer or packing stage is delaying LLM training input",
			Evidence:    []string{fmt.Sprintf("tokenizer_wait=%.1fms data_wait=%.1fms", tr.TokenizerWaitMS, tr.DataWaitMS)},
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
	if tr.AllReduceWaitMS > 50 {
		signals = append(signals, model.Signal{
			Name:        "allreduce_bottleneck",
			Severity:    model.SeverityWarning,
			ScoreImpact: 18,
			Description: "Gradient all-reduce is consuming a large part of step time",
			Evidence:    []string{fmt.Sprintf("all_reduce_wait=%.1fms step=%.1fms world_size=%d", tr.AllReduceWaitMS, tr.StepTimeMS, tr.WorldSize)},
			Timestamp:   now,
		})
	}
	if tr.PipelineBubbleMS > 0 && tr.StepTimeMS > 0 && tr.PipelineBubbleMS/tr.StepTimeMS > 0.20 {
		signals = append(signals, model.Signal{
			Name:        "pipeline_bubble",
			Severity:    model.SeverityWarning,
			ScoreImpact: 15,
			Description: "Pipeline stages are spending too much time idle",
			Evidence:    []string{fmt.Sprintf("pipeline_bubble=%.1fms step=%.1fms stages=%d", tr.PipelineBubbleMS, tr.StepTimeMS, tr.PipelineStages)},
			Timestamp:   now,
		})
	}
	if tr.CheckpointMS > 0 && tr.StepTimeMS > 0 && tr.CheckpointMS/tr.StepTimeMS > 0.30 {
		signals = append(signals, model.Signal{
			Name:        "checkpoint_stall",
			Severity:    model.SeverityWarning,
			ScoreImpact: 14,
			Description: "Checkpointing is stalling training progress",
			Evidence:    []string{fmt.Sprintf("checkpoint=%.1fms step=%.1fms", tr.CheckpointMS, tr.StepTimeMS)},
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
	avgTokens := stream.AvgTokensPerSec(frames)
	if avgTokens > 0 && tr.TokensPerSec > 0 && tr.TokensPerSec < avgTokens*0.65 {
		signals = append(signals, model.Signal{
			Name:        "token_throughput_collapse",
			Severity:    model.SeverityWarning,
			ScoreImpact: 17,
			Description: "Token throughput has dropped sharply against recent history",
			Evidence:    []string{fmt.Sprintf("tokens_per_sec=%.0f avg_tokens_per_sec=%.0f", tr.TokensPerSec, avgTokens)},
			Timestamp:   now,
		})
	}
	if tr.TokensPerSec > 0 && tr.MFU < 0.3 {
		signals = append(signals, model.Signal{
			Name:        "low_mfu",
			Severity:    model.SeverityWarning,
			ScoreImpact: 20,
			Description: "Model FLOPs Utilization is low, indicating inefficient compute usage",
			Evidence:    []string{fmt.Sprintf("mfu=%.1f%% tokens_per_sec=%.0f", tr.MFU*100, tr.TokensPerSec)},
			Timestamp:   now,
		})
	}
	if tr.MemBandwidth > 0 && tr.MemBandwidth > 0.95 {
		signals = append(signals, model.Signal{
			Name:        "memory_bandwidth_limit",
			Severity:    model.SeverityWarning,
			ScoreImpact: 18,
			Description: "Memory bandwidth utilization is high, potentially bottlenecking computation",
			Evidence:    []string{fmt.Sprintf("mem_bandwidth=%.1f%%", tr.MemBandwidth*100)},
			Timestamp:   now,
		})
	}
	if tr.MaxSeqLen > 0 && tr.AvgSeqLen > 0 {
		paddingRatio := 1.0 - (tr.AvgSeqLen / float64(tr.MaxSeqLen))
		if paddingRatio > 0.5 {
			signals = append(signals, model.Signal{
				Name:        "excessive_padding",
				Severity:    model.SeverityWarning,
				ScoreImpact: 16,
				Description: "High sequence padding ratio wasting computation",
				Evidence:    []string{fmt.Sprintf("padding_ratio=%.1f%% avg_len=%.0f max_len=%d", paddingRatio*100, tr.AvgSeqLen, tr.MaxSeqLen)},
				Timestamp:   now,
			})
		}
	}
	if len(tr.Ranks) >= 2 {
		minStep, maxStep := tr.Ranks[0].StepTimeMS, tr.Ranks[0].StepTimeMS
		minRank, maxRank := tr.Ranks[0].Rank, tr.Ranks[0].Rank
		for _, rank := range tr.Ranks[1:] {
			if rank.StepTimeMS < minStep {
				minStep = rank.StepTimeMS
				minRank = rank.Rank
			}
			if rank.StepTimeMS > maxStep {
				maxStep = rank.StepTimeMS
				maxRank = rank.Rank
			}
		}
		if minStep > 0 && maxStep/minStep > 1.35 {
			signals = append(signals, model.Signal{
				Name:        "rank_straggler",
				Severity:    model.SeverityWarning,
				ScoreImpact: 18,
				Description: "One or more ranks are lagging behind the rest of the job",
				Evidence:    []string{fmt.Sprintf("slow_rank=%d %.1fms fast_rank=%d %.1fms", maxRank, maxStep, minRank, minStep)},
				Timestamp:   now,
			})
		}
	}
	return signals
}

func ratio(n, d float64) float64 {
	if d == 0 {
		return 0
	}
	return n / d
}
