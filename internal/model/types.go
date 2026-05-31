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
	StepTimeMS float64   `json:"step_time_ms"`
	Throughput float64   `json:"throughput"`
	DataWaitMS float64   `json:"data_wait_ms"`
	SyncWaitMS float64   `json:"sync_wait_ms"`
	BatchSize  int       `json:"batch_size"`
	GlobalStep int64     `json:"global_step"`
	Timestamp  time.Time `json:"timestamp"`
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
}
