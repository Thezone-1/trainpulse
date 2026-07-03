package metrics

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"sort"
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
		{Metric: ns + ".collect_errors_total", Value: float64(snap.CollectErrors), Type: "counter"},
		{Metric: ns + ".simulated", Value: boolValue(snap.Simulated), Type: "gauge", Tags: collectorTags(snap.Collector)},
		{Metric: ns + ".recommendations_active", Value: float64(len(snap.Recommendations)), Type: "gauge"},
	}
	if u := snap.Utilization; u != nil {
		points = append(points,
			Point{Metric: ns + ".cluster.efficiency_score", Value: u.EfficiencyScore, Type: "gauge"},
			Point{Metric: ns + ".cluster.gpu_util_avg", Value: u.GPUUtilAvg, Type: "gauge"},
			Point{Metric: ns + ".cluster.memory_used_ratio", Value: u.GPUMemUsedRatio, Type: "gauge"},
			Point{Metric: ns + ".cluster.compute_waste_percent", Value: u.ComputeWastePct, Type: "gauge"},
		)
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

// promSample is one value of a metric family: a label set plus a value.
type promSample struct {
	labels map[string]string
	value  float64
}

// promFamily groups every sample that shares a metric name so the exposition
// emits # HELP and # TYPE exactly once per family, as the Prometheus text
// format requires. Emitting them per sample renders the scrape unparsable
// whenever a family has more than one series (multiple GPUs, signals).
type promFamily struct {
	name    string
	help    string
	kind    string
	samples []promSample
}

type promWriter struct {
	order    []string
	families map[string]*promFamily
}

func newPromWriter() *promWriter {
	return &promWriter{families: map[string]*promFamily{}}
}

func (p *promWriter) add(name, help, kind string, value float64, labels map[string]string) {
	family, ok := p.families[name]
	if !ok {
		family = &promFamily{name: name, help: help, kind: kind}
		p.families[name] = family
		p.order = append(p.order, name)
	}
	family.samples = append(family.samples, promSample{labels: labels, value: value})
}

func (p *promWriter) writeTo(w io.Writer) error {
	var b bytes.Buffer
	for _, name := range p.order {
		family := p.families[name]
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s %s\n", family.name, family.help, family.name, family.kind)
		for _, sample := range family.samples {
			fmt.Fprintf(&b, "%s%s %g\n", family.name, promLabels(sample.labels), sample.value)
		}
	}
	_, err := w.Write(b.Bytes())
	return err
}

func WritePrometheus(w io.Writer, snap model.Snapshot, namespace string) error {
	ns := metricName(namespace)
	p := newPromWriter()
	p.add(ns+"_health_score", "Training health score from 0 to 100.", "gauge", snap.Health, nil)
	p.add(ns+"_samples_total", "Telemetry samples processed.", "counter", float64(snap.SampleCount), nil)
	p.add(ns+"_signals_active", "Active diagnostic signals.", "gauge", float64(len(snap.Signals)), nil)
	p.add(ns+"_collect_errors_total", "Telemetry collection failures.", "counter", float64(snap.CollectErrors), nil)
	p.add(ns+"_simulated", "1 when telemetry comes from the simulator instead of real hardware.", "gauge", boolValue(snap.Simulated), collectorTags(snap.Collector))
	p.add(ns+"_recommendations_active", "Active optimization recommendations.", "gauge", float64(len(snap.Recommendations)), nil)
	if u := snap.Utilization; u != nil {
		p.add(ns+"_cluster_efficiency_score", "Composite resource-utilization score, 0 to 100.", "gauge", u.EfficiencyScore, nil)
		p.add(ns+"_cluster_gpu_util_avg", "Average GPU utilization across the cluster.", "gauge", u.GPUUtilAvg, nil)
		p.add(ns+"_cluster_memory_used_ratio", "Aggregate GPU memory used ratio.", "gauge", u.GPUMemUsedRatio, nil)
		p.add(ns+"_cluster_compute_waste_percent", "Idle GPU compute as a percentage.", "gauge", u.ComputeWastePct, nil)
	}
	for _, rec := range snap.Recommendations {
		p.add(ns+"_recommendation_active", "Active recommendation by id and category.", "gauge", 1, map[string]string{
			"id":       rec.ID,
			"category": rec.Category,
		})
	}
	if tr := snap.Telemetry.Training; tr != nil {
		labels := map[string]string{
			"workload_kind": tr.WorkloadKind,
			"model_family":  tr.ModelFamily,
			"model_name":    tr.ModelName,
			"framework":     tr.Framework,
		}
		p.add(ns+"_training_step_time_ms", "Training step time in milliseconds.", "gauge", tr.StepTimeMS, labels)
		p.add(ns+"_training_tokens_per_second", "Training token throughput.", "gauge", tr.TokensPerSec, labels)
		p.add(ns+"_training_mfu", "Model FLOPs utilization, 0 to 1.", "gauge", tr.MFU, labels)
		p.add(ns+"_training_tflops", "Achieved TFLOPs.", "gauge", tr.TFLOPs, labels)
		p.add(ns+"_training_data_wait_ms", "Batch data wait time.", "gauge", tr.DataWaitMS, labels)
		p.add(ns+"_training_tokenizer_wait_ms", "Tokenizer or packing wait time.", "gauge", tr.TokenizerWaitMS, labels)
		p.add(ns+"_training_all_reduce_wait_ms", "All-reduce wait time.", "gauge", tr.AllReduceWaitMS, labels)
		p.add(ns+"_training_pipeline_bubble_ms", "Pipeline parallel idle time.", "gauge", tr.PipelineBubbleMS, labels)
		p.add(ns+"_training_checkpoint_ms", "Checkpoint write time.", "gauge", tr.CheckpointMS, labels)
	}
	for _, gpu := range snap.Telemetry.GPUs {
		labels := map[string]string{"gpu": fmt.Sprint(gpu.Index), "name": gpu.Name}
		p.add(ns+"_gpu_utilization_percent", "GPU utilization percentage.", "gauge", gpu.Utilization, labels)
		p.add(ns+"_gpu_memory_used_mb", "GPU memory used in MiB.", "gauge", float64(gpu.MemoryUsed), labels)
		p.add(ns+"_gpu_memory_total_mb", "GPU memory total in MiB.", "gauge", float64(gpu.MemoryTotal), labels)
		p.add(ns+"_gpu_temperature_celsius", "GPU temperature.", "gauge", gpu.Temperature, labels)
		p.add(ns+"_gpu_power_watts", "GPU power draw.", "gauge", gpu.PowerWatts, labels)
	}
	for _, signal := range snap.Signals {
		p.add(ns+"_signal_active", "Active signal by name and severity.", "gauge", 1, map[string]string{
			"name":     signal.Name,
			"severity": string(signal.Severity),
		})
	}
	return p.writeTo(w)
}

func EncodeJSON(w io.Writer, snap model.Snapshot, namespace string) error {
	return json.NewEncoder(w).Encode(map[string]any{
		"series":   JSONPoints(snap, namespace),
		"snapshot": snap,
	})
}

func boolValue(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

func collectorTags(name string) map[string]string {
	if name == "" {
		return nil
	}
	return map[string]string{"collector": name}
}

func promLabels(labels map[string]string) string {
	if len(labels) == 0 {
		return ""
	}
	keys := make([]string, 0, len(labels))
	for k := range labels {
		if labels[k] == "" {
			continue
		}
		keys = append(keys, k)
	}
	if len(keys) == 0 {
		return ""
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf(`%s="%s"`, metricName(k), escapeLabel(labels[k])))
	}
	return "{" + strings.Join(parts, ",") + "}"
}

func escapeLabel(v string) string {
	v = strings.ReplaceAll(v, `\`, `\\`)
	v = strings.ReplaceAll(v, "\n", `\n`)
	return strings.ReplaceAll(v, `"`, `\"`)
}

var invalidMetric = regexp.MustCompile(`[^a-zA-Z0-9_:]`)

func metricName(s string) string {
	if s == "" {
		return "trainpulse"
	}
	return invalidMetric.ReplaceAllString(s, "_")
}
