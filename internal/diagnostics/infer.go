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
	return out
}
