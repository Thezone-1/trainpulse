package anomaly

import (
	"fmt"
	"strings"

	"github.com/somoprovo/trainpulse/internal/config"
	"github.com/somoprovo/trainpulse/internal/model"
)

type RuleDetector struct {
	rules []config.Rule
}

func NewRuleDetector(rules []config.Rule) *RuleDetector {
	return &RuleDetector{rules: rules}
}

func (r *RuleDetector) Name() string { return "config_rules" }

func (r *RuleDetector) Detect(frames []model.TelemetryFrame) []model.Signal {
	if len(frames) == 0 || len(r.rules) == 0 {
		return nil
	}
	latest := frames[len(frames)-1]
	var signals []model.Signal
	for _, rule := range r.rules {
		values := fieldValues(latest, rule.Field)
		for _, value := range values {
			if !matches(rule.Operator, value, rule.Value) {
				continue
			}
			description := rule.Description
			if description == "" {
				description = fmt.Sprintf("%s %s %.2f", rule.Field, rule.Operator, rule.Value)
			}
			signals = append(signals, model.Signal{
				Name:        rule.Name,
				Severity:    severity(rule.Severity),
				ScoreImpact: scoreImpact(rule.ScoreImpact),
				Description: description,
				Evidence:    []string{fmt.Sprintf("%s=%.2f threshold=%.2f op=%s", rule.Field, value, rule.Value, rule.Operator)},
				Timestamp:   latest.Timestamp,
			})
		}
	}
	return signals
}

func fieldValues(frame model.TelemetryFrame, field string) []float64 {
	tr := frame.Training
	switch strings.ToLower(field) {
	case "host.cpu_utilization":
		return []float64{frame.Host.CPUUtilization}
	case "host.memory_used_mb":
		return []float64{float64(frame.Host.MemoryUsedMB)}
	case "host.load_1":
		return []float64{frame.Host.Load1}
	case "gpu.utilization":
		return gpuValues(frame.GPUs, func(g model.GPUSample) float64 { return g.Utilization })
	case "gpu.memory_used_mb":
		return gpuValues(frame.GPUs, func(g model.GPUSample) float64 { return float64(g.MemoryUsed) })
	case "gpu.memory_used_ratio":
		return gpuValues(frame.GPUs, func(g model.GPUSample) float64 {
			if g.MemoryTotal == 0 {
				return 0
			}
			return float64(g.MemoryUsed) / float64(g.MemoryTotal)
		})
	case "gpu.temperature_c":
		return gpuValues(frame.GPUs, func(g model.GPUSample) float64 { return g.Temperature })
	case "gpu.power_watts":
		return gpuValues(frame.GPUs, func(g model.GPUSample) float64 { return g.PowerWatts })
	}
	if tr == nil {
		return nil
	}
	switch strings.ToLower(field) {
	case "training.step_time_ms":
		return []float64{tr.StepTimeMS}
	case "training.throughput":
		return []float64{tr.Throughput}
	case "training.tokens_per_sec":
		return []float64{tr.TokensPerSec}
	case "training.mfu":
		return []float64{tr.MFU}
	case "training.tflops":
		return []float64{tr.TFLOPs}
	case "training.mem_bandwidth_util":
		return []float64{tr.MemBandwidth}
	case "training.avg_seq_len":
		return []float64{tr.AvgSeqLen}
	case "training.max_seq_len":
		return []float64{float64(tr.MaxSeqLen)}
	case "training.data_wait_ms":
		return []float64{tr.DataWaitMS}
	case "training.tokenizer_wait_ms":
		return []float64{tr.TokenizerWaitMS}
	case "training.sync_wait_ms":
		return []float64{tr.SyncWaitMS}
	case "training.all_reduce_wait_ms":
		return []float64{tr.AllReduceWaitMS}
	case "training.pipeline_bubble_ms":
		return []float64{tr.PipelineBubbleMS}
	case "training.checkpoint_ms":
		return []float64{tr.CheckpointMS}
	}
	return nil
}

func gpuValues(gpus []model.GPUSample, fn func(model.GPUSample) float64) []float64 {
	out := make([]float64, 0, len(gpus))
	for _, gpu := range gpus {
		out = append(out, fn(gpu))
	}
	return out
}

func matches(op string, value, threshold float64) bool {
	switch strings.ToLower(op) {
	case "lt", "<":
		return value < threshold
	case "lte", "<=":
		return value <= threshold
	case "gt", ">":
		return value > threshold
	case "gte", ">=":
		return value >= threshold
	case "eq", "==":
		return value == threshold
	case "neq", "!=":
		return value != threshold
	default:
		return false
	}
}

func severity(v string) model.Severity {
	switch strings.ToLower(v) {
	case "critical":
		return model.SeverityCritical
	case "warning", "warn":
		return model.SeverityWarning
	default:
		return model.SeverityInfo
	}
}

func scoreImpact(v float64) float64 {
	if v <= 0 {
		return 5
	}
	return v
}
