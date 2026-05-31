package metrics

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"strings"

	"github.com/somoprovo/trainpulse/internal/model"
)

type Point struct {
	Metric string            `json:"metric"`
	Value  float64           `json:"value"`
	Type   string            `json:"type"`
	Tags   map[string]string `json:"tags,omitempty"`
}

func JSONPoints(snap model.Snapshot, namespace string) []Point {
	ns := metricName(namespace)
	points := []Point{
		{Metric: ns + ".health_score", Value: snap.Health, Type: "gauge"},
		{Metric: ns + ".samples_total", Value: float64(snap.SampleCount), Type: "counter"},
		{Metric: ns + ".signals_active", Value: float64(len(snap.Signals)), Type: "gauge"},
	}
	if tr := snap.Telemetry.Training; tr != nil {
		tags := map[string]string{
			"workload_kind": tr.WorkloadKind,
			"model_family":  tr.ModelFamily,
			"model_name":    tr.ModelName,
			"framework":     tr.Framework,
		}
		points = append(points,
			Point{Metric: ns + ".training.step_time_ms", Value: tr.StepTimeMS, Type: "gauge", Tags: tags},
			Point{Metric: ns + ".training.tokens_per_second", Value: tr.TokensPerSec, Type: "gauge", Tags: tags},
			Point{Metric: ns + ".training.mfu", Value: tr.MFU, Type: "gauge", Tags: tags},
			Point{Metric: ns + ".training.tflops", Value: tr.TFLOPs, Type: "gauge", Tags: tags},
			Point{Metric: ns + ".training.data_wait_ms", Value: tr.DataWaitMS, Type: "gauge", Tags: tags},
			Point{Metric: ns + ".training.tokenizer_wait_ms", Value: tr.TokenizerWaitMS, Type: "gauge", Tags: tags},
			Point{Metric: ns + ".training.all_reduce_wait_ms", Value: tr.AllReduceWaitMS, Type: "gauge", Tags: tags},
			Point{Metric: ns + ".training.pipeline_bubble_ms", Value: tr.PipelineBubbleMS, Type: "gauge", Tags: tags},
			Point{Metric: ns + ".training.checkpoint_ms", Value: tr.CheckpointMS, Type: "gauge", Tags: tags},
		)
	}
	for _, gpu := range snap.Telemetry.GPUs {
		tags := map[string]string{"gpu": fmt.Sprint(gpu.Index), "name": gpu.Name}
		points = append(points,
			Point{Metric: ns + ".gpu.utilization_percent", Value: gpu.Utilization, Type: "gauge", Tags: tags},
			Point{Metric: ns + ".gpu.memory_used_mb", Value: float64(gpu.MemoryUsed), Type: "gauge", Tags: tags},
			Point{Metric: ns + ".gpu.memory_total_mb", Value: float64(gpu.MemoryTotal), Type: "gauge", Tags: tags},
			Point{Metric: ns + ".gpu.temperature_celsius", Value: gpu.Temperature, Type: "gauge", Tags: tags},
			Point{Metric: ns + ".gpu.power_watts", Value: gpu.PowerWatts, Type: "gauge", Tags: tags},
		)
	}
	for _, signal := range snap.Signals {
		points = append(points, Point{
			Metric: ns + ".signal.active",
			Value:  1,
			Type:   "gauge",
			Tags: map[string]string{
				"name":     signal.Name,
				"severity": string(signal.Severity),
			},
		})
	}
	return points
}

func WritePrometheus(w io.Writer, snap model.Snapshot, namespace string) error {
	var b bytes.Buffer
	ns := metricName(namespace)
	writeMetric(&b, ns+"_health_score", "Training health score from 0 to 100.", snap.Health, nil)
	writeMetric(&b, ns+"_samples_total", "Telemetry samples processed.", float64(snap.SampleCount), nil)
	writeMetric(&b, ns+"_signals_active", "Active diagnostic signals.", float64(len(snap.Signals)), nil)
	if tr := snap.Telemetry.Training; tr != nil {
		labels := map[string]string{
			"workload_kind": tr.WorkloadKind,
			"model_family":  tr.ModelFamily,
			"model_name":    tr.ModelName,
			"framework":     tr.Framework,
		}
		writeMetric(&b, ns+"_training_step_time_ms", "Training step time in milliseconds.", tr.StepTimeMS, labels)
		writeMetric(&b, ns+"_training_tokens_per_second", "Training token throughput.", tr.TokensPerSec, labels)
		writeMetric(&b, ns+"_training_mfu", "Model FLOPs utilization, 0 to 1.", tr.MFU, labels)
		writeMetric(&b, ns+"_training_tflops", "Achieved TFLOPs.", tr.TFLOPs, labels)
		writeMetric(&b, ns+"_training_data_wait_ms", "Batch data wait time.", tr.DataWaitMS, labels)
		writeMetric(&b, ns+"_training_tokenizer_wait_ms", "Tokenizer or packing wait time.", tr.TokenizerWaitMS, labels)
		writeMetric(&b, ns+"_training_all_reduce_wait_ms", "All-reduce wait time.", tr.AllReduceWaitMS, labels)
	}
	for _, gpu := range snap.Telemetry.GPUs {
		labels := map[string]string{"gpu": fmt.Sprint(gpu.Index), "name": gpu.Name}
		writeMetric(&b, ns+"_gpu_utilization_percent", "GPU utilization percentage.", gpu.Utilization, labels)
		writeMetric(&b, ns+"_gpu_memory_used_mb", "GPU memory used in MiB.", float64(gpu.MemoryUsed), labels)
		writeMetric(&b, ns+"_gpu_temperature_celsius", "GPU temperature.", gpu.Temperature, labels)
		writeMetric(&b, ns+"_gpu_power_watts", "GPU power draw.", gpu.PowerWatts, labels)
	}
	for _, signal := range snap.Signals {
		writeMetric(&b, ns+"_signal_active", "Active signal by name and severity.", 1, map[string]string{
			"name":     signal.Name,
			"severity": string(signal.Severity),
		})
	}
	_, err := w.Write(b.Bytes())
	return err
}

func EncodeJSON(w io.Writer, snap model.Snapshot, namespace string) error {
	return json.NewEncoder(w).Encode(map[string]any{
		"series":   JSONPoints(snap, namespace),
		"snapshot": snap,
	})
}

func writeMetric(w io.Writer, name string, help string, value float64, labels map[string]string) {
	fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s gauge\n%s%s %g\n", name, help, name, name, promLabels(labels), value)
}

func promLabels(labels map[string]string) string {
	if len(labels) == 0 {
		return ""
	}
	parts := make([]string, 0, len(labels))
	for k, v := range labels {
		if v == "" {
			continue
		}
		parts = append(parts, fmt.Sprintf(`%s="%s"`, metricName(k), strings.ReplaceAll(v, `"`, `\"`)))
	}
	if len(parts) == 0 {
		return ""
	}
	return "{" + strings.Join(parts, ",") + "}"
}

var invalidMetric = regexp.MustCompile(`[^a-zA-Z0-9_:]`)

func metricName(s string) string {
	if s == "" {
		return "trainpulse"
	}
	return invalidMetric.ReplaceAllString(s, "_")
}
