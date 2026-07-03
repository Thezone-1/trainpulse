package integrations

type Integration struct {
	Name      string   `json:"name"`
	Category  string   `json:"category"`
	Formats   []string `json:"formats"`
	Endpoints []string `json:"endpoints"`
	Notes     string   `json:"notes"`
}

func Catalog() []Integration {
	return []Integration{
		{Name: "Prometheus", Category: "metrics", Formats: []string{"Prometheus text/OpenMetrics"}, Endpoints: []string{"/metrics"}, Notes: "Scrape TrainPulse directly and use Alertmanager for alerts."},
		{Name: "Grafana", Category: "dashboard", Formats: []string{"Prometheus"}, Endpoints: []string{"/metrics"}, Notes: "Use Prometheus, Mimir, or compatible backends as the Grafana data source."},
		{Name: "Datadog", Category: "metrics/events", Formats: []string{"OpenMetrics", "JSON"}, Endpoints: []string{"/metrics", "/v1/metrics", "/v1/events"}, Notes: "Use Datadog Agent OpenMetrics checks or a lightweight custom check for JSON events."},
		{Name: "New Relic", Category: "metrics/events", Formats: []string{"Prometheus", "JSON"}, Endpoints: []string{"/metrics", "/v1/events"}, Notes: "Scrape Prometheus metrics or forward diagnostic events through an agent."},
		{Name: "Elastic/Logstash/Kibana", Category: "logs/events", Formats: []string{"NDJSON", "JSON"}, Endpoints: []string{"/v1/events?format=ndjson"}, Notes: "Ingest diagnostic events as structured logs."},
		{Name: "Splunk", Category: "logs/events", Formats: []string{"NDJSON", "JSON"}, Endpoints: []string{"/v1/events?format=ndjson"}, Notes: "Forward events through Splunk HEC or file tailing."},
		{Name: "OpenTelemetry Collector", Category: "collector", Formats: []string{"Prometheus", "JSON"}, Endpoints: []string{"/metrics", "/v1/events"}, Notes: "Use Prometheus receiver now; OTLP native export can be added without changing detector logic."},
		{Name: "NVIDIA DCGM Exporter", Category: "gpu telemetry peer", Formats: []string{"Prometheus"}, Endpoints: []string{"/metrics"}, Notes: "TrainPulse complements DCGM by adding training-aware diagnosis; it does not replace DCGM."},
		{Name: "Weights & Biases", Category: "experiment tracking", Formats: []string{"JSON metrics/events"}, Endpoints: []string{"/v1/metrics", "/v1/events"}, Notes: "Training code can log TrainPulse health/signals alongside experiment metrics."},
		{Name: "MLflow", Category: "experiment tracking", Formats: []string{"JSON metrics/events"}, Endpoints: []string{"/v1/metrics", "/v1/events"}, Notes: "Log TrainPulse health and diagnosis summaries as run metrics/artifacts."},
		{Name: "Arize Phoenix", Category: "LLM/ML observability", Formats: []string{"OpenTelemetry path", "JSON"}, Endpoints: []string{"/v1/events"}, Notes: "Use TrainPulse events as infrastructure/training context beside traces/evals."},
		{Name: "Langfuse", Category: "LLM observability", Formats: []string{"OpenTelemetry path", "JSON"}, Endpoints: []string{"/v1/events"}, Notes: "Attach TrainPulse events to traces/runs as external observations."},
		{Name: "LangSmith", Category: "LLM observability", Formats: []string{"JSON"}, Endpoints: []string{"/v1/events"}, Notes: "Use as run metadata or external diagnostics beside LangChain traces."},
		{Name: "Helicone/Portkey", Category: "LLM gateway observability", Formats: []string{"JSON"}, Endpoints: []string{"/v1/events"}, Notes: "Gateway tools cover inference traffic; TrainPulse covers training/server health."},
	}
}
