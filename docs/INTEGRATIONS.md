# TrainPulse Integrations

TrainPulse should not replace alerting, dashboarding, tracing, or experiment tracking platforms. It should emit training-aware diagnostics in formats those systems already understand.

## Compatibility Strategy

| Tool family | Examples | TrainPulse output |
| --- | --- | --- |
| Metrics and alerting | Prometheus, Grafana, Mimir, VictoriaMetrics, Thanos | `/metrics` |
| Commercial observability | Datadog, New Relic, Elastic, Splunk | `/metrics`, `/v1/metrics`, `/v1/events`, `/v1/events?format=ndjson` |
| GPU infrastructure monitoring | NVIDIA DCGM Exporter, NVIDIA GPU Operator | complementary Prometheus metrics |
| LLM observability | Langfuse, LangSmith, Helicone, Portkey, Arize Phoenix, Braintrust | `/v1/events` as run context beside traces/evals |
| Experiment tracking | Weights & Biases, MLflow, Neptune, Comet ML, ClearML, TensorBoard | `/v1/metrics` and `/v1/events` logged by the training process |
| Open telemetry pipelines | OpenTelemetry Collector, OpenObserve, SigNoz | scrape `/metrics`; forward `/v1/events` as structured logs |

## Endpoints

- `GET /metrics`: Prometheus/OpenMetrics-compatible metrics.
- `GET /v1/metrics`: JSON metric series for custom agents and Datadog checks.
- `GET /v1/events`: structured diagnostic events for logs/traces/run metadata.
- `GET /v1/events?format=ndjson`: newline-delimited JSON for log shippers.
- `GET /v1/integrations`: machine-readable compatibility catalog.

## What TrainPulse Adds

Existing tools already collect, store, alert, and visualize data. TrainPulse should add the missing training-aware interpretation:

- root-cause-oriented signals
- LLM training semantics
- rank and distributed-runtime context
- compute-waste estimation in future releases
- safe action recommendations in future releases

## Prometheus and Grafana

```yaml
scrape_configs:
  - job_name: trainpulse
    static_configs:
      - targets: ["127.0.0.1:9876"]
```

Useful metrics:

- `trainpulse_health_score`
- `trainpulse_training_tokens_per_second`
- `trainpulse_training_mfu`
- `trainpulse_training_all_reduce_wait_ms`
- `trainpulse_gpu_utilization_percent`
- `trainpulse_signal_active`

## Datadog

Use either:

- Datadog Agent OpenMetrics check against `/metrics`
- a small custom check that reads `/v1/metrics`
- log/event forwarding from `/v1/events?format=ndjson`

TrainPulse should not send Slack/PagerDuty notifications directly. Datadog monitors can own that.

## LLM Observability Tools

Langfuse, LangSmith, Helicone, Portkey, Phoenix, and Braintrust focus mostly on inference traces, prompts, evaluation, cost, latency, and agent behavior. TrainPulse focuses on Linux server and training runtime health.

The practical integration is to attach TrainPulse diagnostic events to a training run, trace, or experiment:

```json
{
  "type": "diagnosis",
  "name": "rank_straggler",
  "message": "A slow rank can force every other rank to wait at synchronization points.",
  "confidence": 0.77,
  "health_score": 61
}
```

## Experiment Tracking

For W&B, MLflow, Neptune, Comet ML, ClearML, and TensorBoard, the training process can periodically read:

- `/v1/metrics` for numeric values
- `/v1/events` for artifacts, annotations, or run notes

This keeps TrainPulse vendor-neutral.
