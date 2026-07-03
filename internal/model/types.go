package model

import "time"

type GPUSample struct {
	Index       int       `json:"index"`
	Name        string    `json:"name"`
	UUID        string    `json:"uuid"`
	Utilization float64   `json:"utilization"`
	MemoryUsed  uint64    `json:"memory_used_mb"`
	MemoryTotal uint64    `json:"memory_total_mb"`
	Temperature float64   `json:"temperature_c"`
	PowerWatts  float64   `json:"power_watts"`
	SMClockMHz  float64   `json:"sm_clock_mhz"`
	Timestamp   time.Time `json:"timestamp"`
}

type HostSample struct {
	CPUUtilization float64   `json:"cpu_utilization"`
	MemoryUsedMB   uint64    `json:"memory_used_mb"`
	MemoryTotalMB  uint64    `json:"memory_total_mb"`
	Load1          float64   `json:"load_1"`
	Timestamp      time.Time `json:"timestamp"`
}

type TrainingSample struct {
	WorkloadKind string `json:"workload_kind,omitempty"`
	ModelFamily  string `json:"model_family,omitempty"`
	ModelName    string `json:"model_name,omitempty"`
	Framework    string `json:"framework,omitempty"`
	Precision    string `json:"precision,omitempty"`

	StepTimeMS     float64 `json:"step_time_ms"`
	Throughput     float64 `json:"throughput"`
	TokensPerSec   float64 `json:"tokens_per_sec"`
	MFU            float64 `json:"mfu"`
	TFLOPs         float64 `json:"tflops"`
	MemBandwidth   float64 `json:"mem_bandwidth_util"`
	AvgSeqLen      float64 `json:"avg_seq_len"`
	MaxSeqLen      int     `json:"max_seq_len"`
	BatchSize      int     `json:"batch_size"`
	MicroBatchSize int     `json:"micro_batch_size,omitempty"`
	GradAccumSteps int     `json:"grad_accum_steps,omitempty"`
	GlobalStep     int64   `json:"global_step"`

	DataWaitMS       float64 `json:"data_wait_ms"`
	TokenizerWaitMS  float64 `json:"tokenizer_wait_ms,omitempty"`
	SyncWaitMS       float64 `json:"sync_wait_ms"`
	AllReduceWaitMS  float64 `json:"all_reduce_wait_ms,omitempty"`
	PipelineBubbleMS float64 `json:"pipeline_bubble_ms,omitempty"`
	CheckpointMS     float64 `json:"checkpoint_ms,omitempty"`

	WorldSize          int          `json:"world_size,omitempty"`
	TensorParallelSize int          `json:"tensor_parallel_size,omitempty"`
	PipelineStages     int          `json:"pipeline_stages,omitempty"`
	DataParallelSize   int          `json:"data_parallel_size,omitempty"`
	Ranks              []RankSample `json:"ranks,omitempty"`

	Timestamp time.Time `json:"timestamp"`
}

type RankSample struct {
	Rank         int     `json:"rank"`
	Node         string  `json:"node,omitempty"`
	GPUIndex     int     `json:"gpu_index,omitempty"`
	StepTimeMS   float64 `json:"step_time_ms"`
	TokensPerSec float64 `json:"tokens_per_sec"`
	DataWaitMS   float64 `json:"data_wait_ms,omitempty"`
	SyncWaitMS   float64 `json:"sync_wait_ms,omitempty"`
}

type TelemetryFrame struct {
	Timestamp time.Time       `json:"timestamp"`
	Host      HostSample      `json:"host"`
	GPUs      []GPUSample     `json:"gpus"`
	Training  *TrainingSample `json:"training,omitempty"`
}

type Severity string

const (
	SeverityInfo     Severity = "info"
	SeverityWarning  Severity = "warning"
	SeverityCritical Severity = "critical"
)

type Signal struct {
	Name        string    `json:"name"`
	Severity    Severity  `json:"severity"`
	ScoreImpact float64   `json:"score_impact"`
	Description string    `json:"description"`
	Evidence    []string  `json:"evidence"`
	Timestamp   time.Time `json:"timestamp"`
}

type Diagnosis struct {
	RootCause   string   `json:"root_cause"`
	Confidence  float64  `json:"confidence"`
	Explanation string   `json:"explanation"`
	Actions     []string `json:"actions"`
}

type Snapshot struct {
	Timestamp   time.Time      `json:"timestamp"`
	Health      float64        `json:"health"`
	Status      Severity       `json:"status"`
	Telemetry   TelemetryFrame `json:"telemetry"`
	Signals     []Signal       `json:"signals"`
	Diagnoses   []Diagnosis    `json:"diagnoses"`
	SampleCount int64          `json:"sample_count"`

	// Collector identifies the telemetry source that produced this snapshot.
	// When a fallback is active this changes (e.g. "sim"), so consumers can
	// tell simulated data from real hardware telemetry.
	Collector     string `json:"collector,omitempty"`
	Simulated     bool   `json:"simulated,omitempty"`
	CollectErrors int64  `json:"collect_errors,omitempty"`
	LastError     string `json:"last_error,omitempty"`
}
