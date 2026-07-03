package optimizer

import (
	"testing"
	"time"

	"github.com/somoprovo/trainpulse/internal/model"
)

func frame(utils []float64, memUsed, memTotal uint64, tr *model.TrainingSample) model.TelemetryFrame {
	now := time.Now()
	gpus := make([]model.GPUSample, len(utils))
	for i, u := range utils {
		gpus[i] = model.GPUSample{Index: i, Utilization: u, MemoryUsed: memUsed, MemoryTotal: memTotal, Timestamp: now}
	}
	return model.TelemetryFrame{Timestamp: now, GPUs: gpus, Training: tr}
}

func recommendationIDs(recs []model.Recommendation) map[string]model.Recommendation {
	out := map[string]model.Recommendation{}
	for _, r := range recs {
		out[r.ID] = r
	}
	return out
}

func TestUtilizationSummary(t *testing.T) {
	e := New()
	f := frame([]float64{90, 70}, 18000, 24000, &model.TrainingSample{MFU: 0.40})
	u := e.Utilization([]model.TelemetryFrame{f})
	if u == nil {
		t.Fatal("expected utilization summary")
	}
	if u.GPUCount != 2 || u.GPUUtilAvg != 80.0 || u.GPUUtilMin != 70.0 {
		t.Fatalf("unexpected utilization: %+v", u)
	}
	if u.GPUMemUsedRatio != 0.75 || u.MemoryHeadroomPct != 25.0 {
		t.Fatalf("unexpected memory ratios: %+v", u)
	}
	// MFU at target => mfu term 100; 0.5*100 + 0.35*80 + 0.15*75 = 89.25 -> 89.3
	if u.EfficiencyScore != 89.3 {
		t.Fatalf("unexpected efficiency score: %v", u.EfficiencyScore)
	}
}

func TestUtilizationNoGPUs(t *testing.T) {
	if u := New().Utilization([]model.TelemetryFrame{{Timestamp: time.Now()}}); u != nil {
		t.Fatalf("expected nil utilization without GPUs, got %+v", u)
	}
	if u := New().Utilization(nil); u != nil {
		t.Fatal("expected nil utilization for empty window")
	}
}

func TestRecommendGrowMicroBatchOnHeadroom(t *testing.T) {
	tr := &model.TrainingSample{MFU: 0.25, MicroBatchSize: 4}
	f := frame([]float64{70, 72}, 10000, 24000, tr) // 42% used, 58% headroom
	recs := recommendationIDs(New().Recommend([]model.TelemetryFrame{f}, nil))
	rec, ok := recs["grow_micro_batch"]
	if !ok {
		t.Fatalf("expected grow_micro_batch, got %v", recs)
	}
	if rec.Current != "4" || rec.Suggested != "8" {
		t.Fatalf("expected 4 -> 8, got %q -> %q", rec.Current, rec.Suggested)
	}
	if rec.AutoApplicable {
		t.Fatal("batch geometry must not be auto-applicable")
	}
}

func TestRecommendNothingWhenSaturated(t *testing.T) {
	// High util, high MFU, high memory use: nothing to reclaim.
	tr := &model.TrainingSample{MFU: 0.48, MicroBatchSize: 4}
	f := frame([]float64{95, 93}, 21500, 24000, tr)
	recs := New().Recommend([]model.TelemetryFrame{f}, nil)
	if len(recs) != 0 {
		t.Fatalf("expected no recommendations for a saturated healthy job, got %+v", recs)
	}
}

func TestSignalDrivenRecommendations(t *testing.T) {
	tr := &model.TrainingSample{GradAccumSteps: 8}
	f := frame([]float64{50, 52}, 22000, 24000, tr)
	signals := []model.Signal{
		{Name: "dataloader_starvation"},
		{Name: "memory_pressure"},
		{Name: "allreduce_bottleneck"},
		{Name: "excessive_padding"},
		{Name: "pipeline_bubble"},
		{Name: "rank_straggler"},
	}
	recs := recommendationIDs(New().Recommend([]model.TelemetryFrame{f}, signals))
	for _, id := range []string{
		"increase_dataloader_workers",
		"relieve_memory_pressure",
		"reduce_sync_frequency",
		"enable_sequence_packing",
		"shrink_pipeline_bubble",
		"isolate_straggler_rank",
	} {
		if _, ok := recs[id]; !ok {
			t.Fatalf("expected recommendation %s, got %v", id, recs)
		}
	}
	if !recs["increase_dataloader_workers"].AutoApplicable {
		t.Fatal("dataloader workers should be auto-applicable")
	}
	if recs["reduce_sync_frequency"].Suggested != "16" {
		t.Fatalf("expected grad accum 8 -> 16, got %q", recs["reduce_sync_frequency"].Suggested)
	}
}

func TestLowUtilWithoutInputBottleneck(t *testing.T) {
	f := frame([]float64{40, 45}, 22000, 24000, &model.TrainingSample{MFU: 0.45})
	recs := recommendationIDs(New().Recommend([]model.TelemetryFrame{f}, nil))
	if _, ok := recs["increase_work_per_step"]; !ok {
		t.Fatalf("expected increase_work_per_step, got %v", recs)
	}
	// With a dataloader signal active, the generic advice must yield to the
	// specific one.
	recs = recommendationIDs(New().Recommend([]model.TelemetryFrame{f}, []model.Signal{{Name: "dataloader_starvation"}}))
	if _, ok := recs["increase_work_per_step"]; ok {
		t.Fatal("generic low-util advice should not fire when input bottleneck explains it")
	}
}
