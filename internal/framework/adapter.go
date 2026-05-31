package framework

import (
	"fmt"
	"strings"
	"time"

	"github.com/somoprovo/trainpulse/internal/model"
)

type Adapter interface {
	Name() string
	Normalize(map[string]any) (model.TrainingSample, error)
}

type Registry struct {
	adapters map[string]Adapter
}

func NewRegistry(adapters ...Adapter) *Registry {
	r := &Registry{adapters: map[string]Adapter{}}
	for _, adapter := range adapters {
		r.Register(adapter)
	}
	return r
}

func DefaultRegistry() *Registry {
	return NewRegistry(
		GenericAdapter{NameValue: "generic"},
		GenericAdapter{NameValue: "pytorch"},
		GenericAdapter{NameValue: "deepspeed"},
		GenericAdapter{NameValue: "megatron"},
		GenericAdapter{NameValue: "huggingface"},
	)
}

func (r *Registry) Register(adapter Adapter) {
	r.adapters[strings.ToLower(adapter.Name())] = adapter
}

func (r *Registry) Normalize(name string, payload map[string]any) (model.TrainingSample, error) {
	if name == "" {
		name = "generic"
	}
	adapter, ok := r.adapters[strings.ToLower(name)]
	if !ok {
		return model.TrainingSample{}, fmt.Errorf("unknown framework %q", name)
	}
	return adapter.Normalize(payload)
}

type GenericAdapter struct {
	NameValue string
}

func (g GenericAdapter) Name() string { return g.NameValue }

func (g GenericAdapter) Normalize(payload map[string]any) (model.TrainingSample, error) {
	s := model.TrainingSample{
		WorkloadKind: "llm_training",
		Framework:    g.NameValue,
		Timestamp:    time.Now(),
	}
	s.WorkloadKind = stringValue(payload, "workload_kind", s.WorkloadKind)
	s.ModelFamily = stringValue(payload, "model_family", "")
	s.ModelName = stringValue(payload, "model_name", stringValue(payload, "model", ""))
	s.Precision = stringValue(payload, "precision", "")
	s.StepTimeMS = floatValue(payload, "step_time_ms", "train_step_ms", "iteration_time_ms", "step_ms")
	s.Throughput = floatValue(payload, "throughput", "samples_per_sec", "examples_per_second")
	s.TokensPerSec = floatValue(payload, "tokens_per_sec", "tokens_per_second", "train_tokens_per_second")
	s.MFU = floatValue(payload, "mfu", "model_flops_utilization")
	s.TFLOPs = floatValue(payload, "tflops", "actual_tflops")
	s.MemBandwidth = floatValue(payload, "mem_bandwidth_util", "memory_bandwidth_utilization")
	s.AvgSeqLen = floatValue(payload, "avg_seq_len", "average_sequence_length")
	s.MaxSeqLen = intValue(payload, "max_seq_len", "sequence_length")
	s.BatchSize = intValue(payload, "batch_size", "global_batch_size")
	s.MicroBatchSize = intValue(payload, "micro_batch_size", "micro_batch")
	s.GradAccumSteps = intValue(payload, "grad_accum_steps", "gradient_accumulation_steps")
	s.GlobalStep = int64(intValue(payload, "global_step", "step"))
	s.DataWaitMS = floatValue(payload, "data_wait_ms", "dataloader_ms", "data_loader_ms")
	s.TokenizerWaitMS = floatValue(payload, "tokenizer_wait_ms", "tokenization_ms", "packing_ms")
	s.SyncWaitMS = floatValue(payload, "sync_wait_ms", "sync_ms")
	s.AllReduceWaitMS = floatValue(payload, "all_reduce_wait_ms", "allreduce_ms", "gradient_allreduce_ms")
	s.PipelineBubbleMS = floatValue(payload, "pipeline_bubble_ms", "pipeline_idle_ms")
	s.CheckpointMS = floatValue(payload, "checkpoint_ms", "checkpoint_save_ms")
	s.WorldSize = intValue(payload, "world_size", "num_processes")
	s.TensorParallelSize = intValue(payload, "tensor_parallel_size", "tp_size")
	s.PipelineStages = intValue(payload, "pipeline_stages", "pp_size")
	s.DataParallelSize = intValue(payload, "data_parallel_size", "dp_size")
	return s, nil
}

func stringValue(payload map[string]any, key string, fallback string) string {
	if v, ok := payload[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return fallback
}

func floatValue(payload map[string]any, keys ...string) float64 {
	for _, key := range keys {
		switch v := payload[key].(type) {
		case float64:
			return v
		case int:
			return float64(v)
		case jsonNumber:
			f, _ := v.Float64()
			return f
		}
	}
	return 0
}

func intValue(payload map[string]any, keys ...string) int {
	for _, key := range keys {
		switch v := payload[key].(type) {
		case float64:
			return int(v)
		case int:
			return v
		case jsonNumber:
			i, _ := v.Int64()
			return int(i)
		}
	}
	return 0
}

type jsonNumber interface {
	Float64() (float64, error)
	Int64() (int64, error)
}
