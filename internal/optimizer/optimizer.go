// Package optimizer turns live telemetry into concrete resource-utilization
// recommendations: where the cluster is leaving compute or memory on the
// table, and which knob to turn to reclaim it.
//
// The daemon is advisory by design — an external process cannot safely mutate
// a running job's batch geometry or allocator behavior. Each recommendation
// therefore says whether a cooperating training loop may auto-apply it
// (AutoApplicable) or whether it needs a human decision.
package optimizer

import (
	"fmt"
	"math"

	"github.com/somoprovo/trainpulse/internal/model"
	"github.com/somoprovo/trainpulse/internal/stream"
)

// Thresholds for the utilization heuristics. Exported nowhere: these encode
// the product's opinion of "leaving performance on the table".
const (
	memHeadroomForGrowth = 0.35 // free GPU memory ratio that justifies bigger batches
	lowUtilPct           = 60   // avg GPU util below this is under-feeding
	goodMFU              = 0.40
	imbalanceSpreadPct   = 25 // min/max GPU util spread that wastes the slowest GPU
)

type Engine struct{}

func New() *Engine { return &Engine{} }

// Utilization computes the cluster-level resource picture for the latest
// frame, with window averages smoothing single-tick noise.
func (e *Engine) Utilization(frames []model.TelemetryFrame) *model.ClusterUtilization {
	if len(frames) == 0 {
		return nil
	}
	latest := frames[len(frames)-1]
	if len(latest.GPUs) == 0 {
		return nil
	}
	minUtil := math.MaxFloat64
	var sumUtil, memUsed, memTotal float64
	for _, gpu := range latest.GPUs {
		sumUtil += gpu.Utilization
		minUtil = math.Min(minUtil, gpu.Utilization)
		memUsed += float64(gpu.MemoryUsed)
		memTotal += float64(gpu.MemoryTotal)
	}
	avgUtil := sumUtil / float64(len(latest.GPUs))
	memRatio := 0.0
	if memTotal > 0 {
		memRatio = memUsed / memTotal
	}
	u := &model.ClusterUtilization{
		GPUCount:          len(latest.GPUs),
		GPUUtilAvg:        round1(avgUtil),
		GPUUtilMin:        round1(minUtil),
		GPUMemUsedRatio:   round3(memRatio),
		ComputeWastePct:   round1(100 - avgUtil),
		MemoryHeadroomPct: round1((1 - memRatio) * 100),
	}
	if tr := latest.Training; tr != nil {
		u.MFU = tr.MFU
	}
	u.EfficiencyScore = round1(efficiencyScore(u, latest.Training))
	return u
}

// efficiencyScore folds compute use, memory use, and (when known) MFU into a
// single 0-100 "how much of the hardware you paid for is working" number.
// MFU dominates when available because it measures useful work, not busyness.
func efficiencyScore(u *model.ClusterUtilization, tr *model.TrainingSample) float64 {
	compute := u.GPUUtilAvg // 0-100
	memory := u.GPUMemUsedRatio * 100
	if tr != nil && tr.MFU > 0 {
		mfu := math.Min(tr.MFU/goodMFU, 1.0) * 100
		return clamp(0.5*mfu+0.35*compute+0.15*memory, 0, 100)
	}
	return clamp(0.7*compute+0.3*memory, 0, 100)
}

// Recommend derives tuning actions from the telemetry window and the signals
// the anomaly engine already raised this tick.
func (e *Engine) Recommend(frames []model.TelemetryFrame, signals []model.Signal) []model.Recommendation {
	if len(frames) == 0 {
		return nil
	}
	latest := frames[len(frames)-1]
	active := make(map[string]bool, len(signals))
	for _, s := range signals {
		active[s.Name] = true
	}
	util := e.Utilization(frames)
	var recs []model.Recommendation

	recs = append(recs, memoryRecommendations(latest, util, active)...)
	recs = append(recs, computeRecommendations(frames, latest, util, active)...)
	recs = append(recs, dataRecommendations(active)...)
	recs = append(recs, communicationRecommendations(latest, active)...)
	return recs
}

func memoryRecommendations(latest model.TelemetryFrame, util *model.ClusterUtilization, active map[string]bool) []model.Recommendation {
	var recs []model.Recommendation
	tr := latest.Training
	if util == nil {
		return nil
	}
	// Headroom + underused compute: the job can grow into the memory it paid for.
	if util.GPUMemUsedRatio > 0 && (1-util.GPUMemUsedRatio) >= memHeadroomForGrowth && tr != nil && (util.GPUUtilAvg < 85 || tr.MFU > 0 && tr.MFU < goodMFU) {
		current, suggested := "", ""
		param := "micro_batch_size"
		if tr.MicroBatchSize > 0 {
			current = fmt.Sprint(tr.MicroBatchSize)
			suggested = fmt.Sprint(tr.MicroBatchSize * 2)
		}
		recs = append(recs, model.Recommendation{
			ID:         "grow_micro_batch",
			Category:   "memory",
			Parameter:  param,
			Current:    current,
			Suggested:  suggested,
			Impact:     "Raise arithmetic intensity per step; typically the cheapest MFU gain available",
			Confidence: 0.72,
			Rationale: fmt.Sprintf("GPU memory is only %.0f%% used with %.0f%% average utilization — the batch can grow before memory becomes the constraint",
				util.GPUMemUsedRatio*100, util.GPUUtilAvg),
			Evidence:       []string{fmt.Sprintf("mem_used_ratio=%.2f gpu_util_avg=%.1f%%", util.GPUMemUsedRatio, util.GPUUtilAvg)},
			AutoApplicable: false, // batch geometry changes affect convergence; needs the trainer's consent
		})
	}
	if active["memory_pressure"] {
		recs = append(recs, model.Recommendation{
			ID:             "relieve_memory_pressure",
			Category:       "memory",
			Parameter:      "activation_checkpointing",
			Suggested:      "enable",
			Impact:         "Trade ~20-30% recompute for large activation memory savings, avoiding OOM and fragmentation stalls",
			Confidence:     0.78,
			Rationale:      "GPU memory occupancy is above 90%; allocation failure or fragmentation-induced stalls become likely",
			AutoApplicable: false,
		})
	}
	return recs
}

func computeRecommendations(frames []model.TelemetryFrame, latest model.TelemetryFrame, util *model.ClusterUtilization, active map[string]bool) []model.Recommendation {
	var recs []model.Recommendation
	if util == nil {
		return nil
	}
	if len(latest.GPUs) >= 2 && util.GPUUtilAvg-util.GPUUtilMin > imbalanceSpreadPct || active["synchronization_imbalance"] {
		recs = append(recs, model.Recommendation{
			ID:         "rebalance_workload",
			Category:   "compute",
			Impact:     "Every synchronization waits for the slowest GPU; leveling the load recovers the gap on all ranks",
			Confidence: 0.70,
			Rationale: fmt.Sprintf("GPU utilization spread (min %.0f%%, avg %.0f%%) means faster GPUs idle at sync points",
				util.GPUUtilMin, util.GPUUtilAvg),
			Evidence:       []string{fmt.Sprintf("gpu_util_min=%.1f%% gpu_util_avg=%.1f%%", util.GPUUtilMin, util.GPUUtilAvg)},
			AutoApplicable: false,
		})
	}
	if active["excessive_padding"] {
		recs = append(recs, model.Recommendation{
			ID:             "enable_sequence_packing",
			Category:       "compute",
			Parameter:      "sequence_packing",
			Suggested:      "enable length-bucketed packing",
			Impact:         "Stop spending FLOPs on pad tokens; padding above 50% roughly halves useful throughput",
			Confidence:     0.75,
			Rationale:      "Average sequence length is far below max_seq_len, so a large share of compute goes to padding",
			AutoApplicable: false,
		})
	}
	if active["pipeline_bubble"] {
		current, suggested := "", ""
		if tr := latest.Training; tr != nil && tr.GradAccumSteps > 0 {
			current = fmt.Sprint(tr.GradAccumSteps)
			suggested = fmt.Sprint(tr.GradAccumSteps * 2)
		}
		recs = append(recs, model.Recommendation{
			ID:             "shrink_pipeline_bubble",
			Category:       "compute",
			Parameter:      "grad_accum_steps",
			Current:        current,
			Suggested:      suggested,
			Impact:         "More microbatches per flush shrink the pipeline fill/drain bubble proportionally",
			Confidence:     0.71,
			Rationale:      "Pipeline stages spend over 20% of each step idle waiting on adjacent stages",
			AutoApplicable: false,
		})
	}
	// Sustained low utilization without an input-side signal: the step itself
	// is too small for the hardware.
	if stream.AvgGPUUtil(frames) < lowUtilPct && !active["dataloader_starvation"] && !active["tokenizer_bottleneck"] {
		recs = append(recs, model.Recommendation{
			ID:         "increase_work_per_step",
			Category:   "compute",
			Impact:     "Larger per-step work amortizes kernel launch and synchronization overhead",
			Confidence: 0.62,
			Rationale: fmt.Sprintf("Average GPU utilization is %.0f%% with no input bottleneck detected — the kernels themselves are not saturating the device",
				stream.AvgGPUUtil(frames)),
			AutoApplicable: false,
		})
	}
	return recs
}

func dataRecommendations(active map[string]bool) []model.Recommendation {
	var recs []model.Recommendation
	if active["dataloader_starvation"] {
		recs = append(recs, model.Recommendation{
			ID:             "increase_dataloader_workers",
			Category:       "data",
			Parameter:      "dataloader_workers",
			Suggested:      "increase until data_wait_ms stays under 10ms",
			Impact:         "GPUs currently idle waiting for batches; input parallelism reclaims that time directly",
			Confidence:     0.80,
			Rationale:      "High data wait with low GPU utilization means the input pipeline, not compute, sets the step time",
			AutoApplicable: true, // worker count is convergence-neutral
		})
	}
	if active["tokenizer_bottleneck"] {
		recs = append(recs, model.Recommendation{
			ID:             "pretokenize_dataset",
			Category:       "data",
			Parameter:      "tokenization",
			Suggested:      "pre-tokenize offline or add preprocessing workers",
			Impact:         "Move tokenization off the training critical path",
			Confidence:     0.74,
			Rationale:      "Tokenizer wait time is high enough to delay batch delivery",
			AutoApplicable: true,
		})
	}
	return recs
}

func communicationRecommendations(latest model.TelemetryFrame, active map[string]bool) []model.Recommendation {
	var recs []model.Recommendation
	if active["allreduce_bottleneck"] || active["communication_bottleneck"] {
		current, suggested := "", ""
		if tr := latest.Training; tr != nil && tr.GradAccumSteps > 0 {
			current = fmt.Sprint(tr.GradAccumSteps)
			suggested = fmt.Sprint(tr.GradAccumSteps * 2)
		}
		recs = append(recs, model.Recommendation{
			ID:             "reduce_sync_frequency",
			Category:       "communication",
			Parameter:      "grad_accum_steps",
			Current:        current,
			Suggested:      suggested,
			Impact:         "Fewer gradient synchronizations per optimizer step; also verify compute/comm overlap and bucket sizes",
			Confidence:     0.68,
			Rationale:      "Gradient synchronization consumes a large share of step time",
			AutoApplicable: false,
		})
	}
	if active["rank_straggler"] {
		recs = append(recs, model.Recommendation{
			ID:             "isolate_straggler_rank",
			Category:       "communication",
			Impact:         "One slow rank taxes every rank; fixing or cordoning that node recovers cluster-wide throughput",
			Confidence:     0.72,
			Rationale:      "Per-rank step times diverge by more than 35%; the slowest rank gates every collective",
			AutoApplicable: false,
		})
	}
	return recs
}

func clamp(v, lo, hi float64) float64 {
	return math.Min(hi, math.Max(lo, v))
}

func round1(v float64) float64 { return math.Round(v*10) / 10 }

func round3(v float64) float64 { return math.Round(v*1000) / 1000 }
