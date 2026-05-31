package diagnostics

import "github.com/somoprovo/trainpulse/internal/model"

type Inferer struct{}

func New() *Inferer { return &Inferer{} }

func (i *Inferer) Infer(signals []model.Signal) []model.Diagnosis {
	names := map[string]model.Signal{}
	for _, signal := range signals {
		names[signal.Name] = signal
	}
	var out []model.Diagnosis
	if _, ok := names["dataloader_starvation"]; ok {
		out = append(out, model.Diagnosis{
			RootCause:   "dataloader_starvation",
			Confidence:  0.86,
			Explanation: "GPU work is waiting on input batches instead of compute.",
			Actions:     []string{"Increase dataloader workers", "Check storage and network dataset latency", "Enable prefetching or pinned memory"},
		})
	}
	if _, ok := names["synchronization_imbalance"]; ok {
		out = append(out, model.Diagnosis{
			RootCause:   "distributed_sync_imbalance",
			Confidence:  0.78,
			Explanation: "Workers are progressing unevenly, which can stall collective synchronization.",
			Actions:     []string{"Inspect per-rank step time", "Check uneven batch shapes", "Verify network fabric and NCCL logs"},
		})
	}
	if _, ok := names["thermal_instability"]; ok {
		out = append(out, model.Diagnosis{
			RootCause:   "thermal_throttling_risk",
			Confidence:  0.82,
			Explanation: "GPU temperature is high enough to risk clock throttling and throughput loss.",
			Actions:     []string{"Inspect fan and chassis airflow", "Reduce power limit temporarily", "Check neighboring GPU temperatures"},
		})
	}
	if _, ok := names["memory_pressure"]; ok {
		out = append(out, model.Diagnosis{
			RootCause:   "gpu_memory_pressure",
			Confidence:  0.74,
			Explanation: "GPU memory occupancy is near the limit and may lead to fragmentation or allocation failure.",
			Actions:     []string{"Review activation checkpointing", "Lower batch size", "Inspect allocator reserved versus allocated memory"},
		})
	}
	if _, ok := names["communication_bottleneck"]; ok {
		out = append(out, model.Diagnosis{
			RootCause:   "communication_bottleneck",
			Confidence:  0.72,
			Explanation: "Training progress is being delayed by synchronization or communication waits.",
			Actions:     []string{"Check interconnect counters", "Tune bucket sizes", "Verify compute and communication overlap"},
		})
	}
	if _, ok := names["low_mfu"]; ok {
		out = append(out, model.Diagnosis{
			RootCause:   "llm_compute_efficiency_loss",
			Confidence:  0.76,
			Explanation: "Token throughput exists, but model FLOPs utilization is low for the observed training step.",
			Actions:     []string{"Check kernel fusion and precision mode", "Review sequence packing efficiency", "Compare achieved TFLOPs with expected hardware peak"},
		})
	}
	if _, ok := names["tokenizer_bottleneck"]; ok {
		out = append(out, model.Diagnosis{
			RootCause:   "tokenization_or_packing_bottleneck",
			Confidence:  0.80,
			Explanation: "The LLM input path is spending too much time tokenizing, packing, or preparing batches.",
			Actions:     []string{"Pre-tokenize the dataset", "Increase preprocessing workers", "Inspect packing and shuffling latency"},
		})
	}
	if _, ok := names["allreduce_bottleneck"]; ok {
		out = append(out, model.Diagnosis{
			RootCause:   "distributed_gradient_communication",
			Confidence:  0.78,
			Explanation: "Gradient synchronization is taking enough time to reduce useful training throughput.",
			Actions:     []string{"Inspect NCCL logs and fabric counters", "Tune gradient bucket size", "Validate topology-aware rank placement"},
		})
	}
	if _, ok := names["rank_straggler"]; ok {
		out = append(out, model.Diagnosis{
			RootCause:   "rank_straggler",
			Confidence:  0.77,
			Explanation: "A slow rank can force every other rank to wait at synchronization points.",
			Actions:     []string{"Compare per-rank dataloader and step timing", "Check node-local thermal or network issues", "Inspect uneven sequence lengths per rank"},
		})
	}
	if _, ok := names["excessive_padding"]; ok {
		out = append(out, model.Diagnosis{
			RootCause:   "sequence_packing_inefficiency",
			Confidence:  0.73,
			Explanation: "The job is spending compute on padding tokens rather than useful sequence tokens.",
			Actions:     []string{"Use length bucketing", "Improve packed sequence construction", "Track useful tokens versus padded tokens"},
		})
	}
	if _, ok := names["checkpoint_stall"]; ok {
		out = append(out, model.Diagnosis{
			RootCause:   "checkpoint_io_stall",
			Confidence:  0.70,
			Explanation: "Checkpoint writes are large enough to interrupt normal training cadence.",
			Actions:     []string{"Checkpoint asynchronously", "Increase checkpoint interval", "Inspect filesystem and object-store latency"},
		})
	}
	return out
}
