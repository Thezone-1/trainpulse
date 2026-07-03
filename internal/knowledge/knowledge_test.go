package knowledge

import (
	"testing"

	"github.com/somoprovo/trainpulse/internal/config"
	"github.com/somoprovo/trainpulse/internal/model"
)

func signals(names ...string) []model.Signal {
	out := make([]model.Signal, 0, len(names))
	for _, n := range names {
		out = append(out, model.Signal{Name: n})
	}
	return out
}

// goldenDefault mirrors the previously hard-coded diagnostics map, keyed by the
// triggering signal name. It guarantees the data-driven default KB is
// behavior-identical to the original compiled-in logic.
var goldenDefault = map[string]model.Diagnosis{
	"dataloader_starvation": {
		RootCause:   "dataloader_starvation",
		Confidence:  0.86,
		Explanation: "GPU work is waiting on input batches instead of compute.",
		Actions:     []string{"Increase dataloader workers", "Check storage and network dataset latency", "Enable prefetching or pinned memory"},
	},
	"synchronization_imbalance": {
		RootCause:   "distributed_sync_imbalance",
		Confidence:  0.78,
		Explanation: "Workers are progressing unevenly, which can stall collective synchronization.",
		Actions:     []string{"Inspect per-rank step time", "Check uneven batch shapes", "Verify network fabric and NCCL logs"},
	},
	"thermal_instability": {
		RootCause:   "thermal_throttling_risk",
		Confidence:  0.82,
		Explanation: "GPU temperature is high enough to risk clock throttling and throughput loss.",
		Actions:     []string{"Inspect fan and chassis airflow", "Reduce power limit temporarily", "Check neighboring GPU temperatures"},
	},
	"memory_pressure": {
		RootCause:   "gpu_memory_pressure",
		Confidence:  0.74,
		Explanation: "GPU memory occupancy is near the limit and may lead to fragmentation or allocation failure.",
		Actions:     []string{"Review activation checkpointing", "Lower batch size", "Inspect allocator reserved versus allocated memory"},
	},
	"communication_bottleneck": {
		RootCause:   "communication_bottleneck",
		Confidence:  0.72,
		Explanation: "Training progress is being delayed by synchronization or communication waits.",
		Actions:     []string{"Check interconnect counters", "Tune bucket sizes", "Verify compute and communication overlap"},
	},
	"low_mfu": {
		RootCause:   "llm_compute_efficiency_loss",
		Confidence:  0.76,
		Explanation: "Token throughput exists, but model FLOPs utilization is low for the observed training step.",
		Actions:     []string{"Check kernel fusion and precision mode", "Review sequence packing efficiency", "Compare achieved TFLOPs with expected hardware peak"},
	},
	"tokenizer_bottleneck": {
		RootCause:   "tokenization_or_packing_bottleneck",
		Confidence:  0.80,
		Explanation: "The LLM input path is spending too much time tokenizing, packing, or preparing batches.",
		Actions:     []string{"Pre-tokenize the dataset", "Increase preprocessing workers", "Inspect packing and shuffling latency"},
	},
	"allreduce_bottleneck": {
		RootCause:   "distributed_gradient_communication",
		Confidence:  0.78,
		Explanation: "Gradient synchronization is taking enough time to reduce useful training throughput.",
		Actions:     []string{"Inspect NCCL logs and fabric counters", "Tune gradient bucket size", "Validate topology-aware rank placement"},
	},
	"rank_straggler": {
		RootCause:   "rank_straggler",
		Confidence:  0.77,
		Explanation: "A slow rank can force every other rank to wait at synchronization points.",
		Actions:     []string{"Compare per-rank dataloader and step timing", "Check node-local thermal or network issues", "Inspect uneven sequence lengths per rank"},
	},
	"excessive_padding": {
		RootCause:   "sequence_packing_inefficiency",
		Confidence:  0.73,
		Explanation: "The job is spending compute on padding tokens rather than useful sequence tokens.",
		Actions:     []string{"Use length bucketing", "Improve packed sequence construction", "Track useful tokens versus padded tokens"},
	},
	"checkpoint_stall": {
		RootCause:   "checkpoint_io_stall",
		Confidence:  0.70,
		Explanation: "Checkpoint writes are large enough to interrupt normal training cadence.",
		Actions:     []string{"Checkpoint asynchronously", "Increase checkpoint interval", "Inspect filesystem and object-store latency"},
	},
	"gpu_underutilization": {
		RootCause:   "gpu_underutilization",
		Confidence:  0.68,
		Explanation: "GPU compute is idle for a large share of each step, so the job is not bound by the accelerator.",
		Actions:     []string{"Profile the step to find CPU, IO, or launch-overhead gaps", "Increase batch size or fuse small kernels", "Check for host-side preprocessing on the critical path"},
	},
	"pipeline_bubble": {
		RootCause:   "pipeline_parallel_bubble",
		Confidence:  0.72,
		Explanation: "Pipeline stages are waiting on each other instead of computing, wasting accelerator time every step.",
		Actions:     []string{"Increase microbatch count per pipeline flush", "Rebalance layers across pipeline stages", "Consider interleaved pipeline scheduling"},
	},
	"throughput_collapse": {
		RootCause:   "step_time_regression",
		Confidence:  0.71,
		Explanation: "Throughput dropped sharply against the job's own recent history, which usually indicates a new external stall rather than model changes.",
		Actions:     []string{"Compare current step against the last healthy window", "Check for concurrent jobs, checkpoints, or evaluation runs", "Inspect dataloader, network, and storage latency for regressions"},
	},
	"memory_bandwidth_limit": {
		RootCause:   "memory_bandwidth_bound",
		Confidence:  0.69,
		Explanation: "The workload is saturating GPU memory bandwidth, so more compute will not raise throughput until data movement shrinks.",
		Actions:     []string{"Enable or verify kernel fusion to cut intermediate reads/writes", "Review precision and activation checkpointing choices", "Profile with a roofline model to confirm the bandwidth ceiling"},
	},
}

func diagnosisEqual(a, b model.Diagnosis) bool {
	if a.RootCause != b.RootCause || a.Confidence != b.Confidence || a.Explanation != b.Explanation {
		return false
	}
	if len(a.Actions) != len(b.Actions) {
		return false
	}
	for i := range a.Actions {
		if a.Actions[i] != b.Actions[i] {
			return false
		}
	}
	return true
}

func TestDefaultParses(t *testing.T) {
	base := Default()
	if got := len(base.Rules()); got != len(goldenDefault) {
		t.Fatalf("default KB has %d rules, want %d", got, len(goldenDefault))
	}
}

func TestDefaultBehaviorPerSignal(t *testing.T) {
	base := Default()
	for signal, want := range goldenDefault {
		got := base.Infer(signals(signal))
		if len(got) != 1 {
			t.Fatalf("signal %q produced %d diagnoses, want 1", signal, len(got))
		}
		if !diagnosisEqual(got[0], want) {
			t.Errorf("signal %q:\n got %+v\nwant %+v", signal, got[0], want)
		}
	}
}

func TestInferPreservesOrder(t *testing.T) {
	// Feed signals in reverse KB order; output must still follow KB order.
	base := Default()
	got := base.Infer(signals("checkpoint_stall", "dataloader_starvation"))
	if len(got) != 2 {
		t.Fatalf("got %d diagnoses, want 2", len(got))
	}
	if got[0].RootCause != "dataloader_starvation" || got[1].RootCause != "checkpoint_io_stall" {
		t.Errorf("order not preserved: %s then %s", got[0].RootCause, got[1].RootCause)
	}
}

func TestNoSignalsNoDiagnoses(t *testing.T) {
	if got := Default().Infer(nil); len(got) != 0 {
		t.Fatalf("expected no diagnoses, got %d", len(got))
	}
}

func TestOverrideReplacesInPlace(t *testing.T) {
	base := New([]config.DiagnosisRule{{
		RootCause:   "dataloader_starvation",
		WhenSignals: []string{"dataloader_starvation"},
		Confidence:  0.99,
		Explanation: "custom",
		Actions:     []string{"do the thing"},
	}})
	rules := base.Rules()
	if len(rules) != len(goldenDefault) {
		t.Fatalf("override changed rule count to %d, want %d", len(rules), len(goldenDefault))
	}
	// Must remain the first rule (in place), with new values.
	if rules[0].RootCause != "dataloader_starvation" || rules[0].Confidence != 0.99 {
		t.Fatalf("override not applied in place: %+v", rules[0])
	}
	got := base.Infer(signals("dataloader_starvation"))
	if len(got) != 1 || got[0].Confidence != 0.99 || got[0].Explanation != "custom" {
		t.Errorf("override diagnosis not returned: %+v", got)
	}
}

func TestOverrideAppendsNewRootCause(t *testing.T) {
	base := New([]config.DiagnosisRule{{
		RootCause:   "team_custom_cause",
		WhenSignals: []string{"team_low_mfu"},
		Confidence:  0.5,
		Explanation: "team rule",
		Actions:     []string{"page the on-call"},
	}})
	if got := len(base.Rules()); got != len(goldenDefault)+1 {
		t.Fatalf("expected %d rules, got %d", len(goldenDefault)+1, got)
	}
	got := base.Infer(signals("team_low_mfu"))
	if len(got) != 1 || got[0].RootCause != "team_custom_cause" {
		t.Errorf("custom diagnosis not returned: %+v", got)
	}
}

func TestMatchAllRequiresEverySignal(t *testing.T) {
	base := New([]config.DiagnosisRule{{
		RootCause:   "combined_cause",
		WhenSignals: []string{"low_mfu", "excessive_padding"},
		Match:       "all",
		Confidence:  0.6,
		Explanation: "both present",
		Actions:     []string{"investigate packing"},
	}})
	if got := base.Infer(signals("low_mfu")); len(got) != 1 {
		// only the built-in low_mfu diagnosis, not combined_cause
		if got[0].RootCause == "combined_cause" {
			t.Fatalf("match=all fired with only one signal present")
		}
	}
	got := base.Infer(signals("low_mfu", "excessive_padding"))
	found := false
	for _, d := range got {
		if d.RootCause == "combined_cause" {
			found = true
		}
	}
	if !found {
		t.Errorf("match=all did not fire when both signals present: %+v", got)
	}
}
