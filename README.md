# TrainPulse

TrainPulse is a lightweight Go daemon for predictive diagnostics of AI and LLM training systems. It collects runtime telemetry, runs low-latency anomaly checks, scores training health, infers likely root causes, and exposes real-time terminal-native stats.

## First slice

- Linux daemon shape with local HTTP snapshot API.
- NVIDIA GPU telemetry through `nvidia-smi`.
- Host memory and load telemetry from `/proc`.
- Simulation mode for development without GPUs.
- Real-time health scoring and terminal dashboard.
- Early diagnostic rules for dataloader starvation, GPU underutilization, sync imbalance, memory pressure, thermal instability, and throughput collapse.
- LLM-native signals for tokens/sec, MFU, sequence padding, tokenizer stalls, all-reduce waits, checkpoint stalls, pipeline bubbles, and per-rank stragglers.

## Build

```sh
go build ./cmd/trainpulse
```

Release builds stamp the version into the binary:

```sh
make build        # version from git describe
make release      # cross-compiled binaries in dist/
./trainpulse version
```

## Run

Use simulation mode anywhere:

```sh
./trainpulse top -mode sim -interval 1s
```

Run as a daemon on a Linux GPU host:

```sh
./trainpulse daemon -addr 127.0.0.1:9876 -mode auto -interval 1s
```

Run with a config file:

```sh
./trainpulse daemon -config config.example.json
```

Fetch one JSON snapshot:

```sh
curl http://127.0.0.1:9876/v1/snapshot
```

Prometheus scrape endpoint:

```sh
curl http://127.0.0.1:9876/metrics
```

Datadog-friendly JSON metrics:

```sh
curl http://127.0.0.1:9876/v1/metrics
```

Send LLM runtime metrics from a training loop:

```sh
curl -X POST http://127.0.0.1:9876/v1/training \
  -H 'content-type: application/json' \
  -d '{
    "workload_kind": "llm_pretraining",
    "model_family": "llama",
    "model_name": "llama-7b",
    "framework": "pytorch",
    "precision": "bf16",
    "global_step": 1200,
    "step_time_ms": 184.2,
    "tokens_per_sec": 72500,
    "mfu": 0.42,
    "tflops": 260.4,
    "avg_seq_len": 1800,
    "max_seq_len": 2048,
    "data_wait_ms": 12.4,
    "tokenizer_wait_ms": 3.0,
    "sync_wait_ms": 18.0,
    "all_reduce_wait_ms": 16.0,
    "world_size": 8,
    "ranks": [
      {"rank": 0, "gpu_index": 0, "step_time_ms": 184.2, "tokens_per_sec": 9100},
      {"rank": 1, "gpu_index": 1, "step_time_ms": 188.0, "tokens_per_sec": 8900}
    ]
  }'
```

Send framework-style metrics and let TrainPulse normalize them:

```sh
curl -X POST 'http://127.0.0.1:9876/v1/framework?name=deepspeed' \
  -H 'content-type: application/json' \
  -d '{
    "model": "llama-7b",
    "train_tokens_per_second": 72500,
    "model_flops_utilization": 0.42,
    "gradient_allreduce_ms": 16.0,
    "global_batch_size": 512,
    "gradient_accumulation_steps": 16
  }'
```

Supported adapter names today: `generic`, `pytorch`, `deepspeed`, `megatron`, `huggingface`.

## Resource optimization

TrainPulse continuously scores how much of the cluster's compute and memory
the job actually uses, and recommends concrete knobs to reclaim the rest:

```sh
curl http://127.0.0.1:9876/v1/recommendations
```

```json
{
  "utilization": {
    "gpu_count": 8, "gpu_util_avg": 71.0, "gpu_mem_used_ratio": 0.42,
    "compute_waste_pct": 29.0, "memory_headroom_pct": 58.0,
    "mfu": 0.31, "efficiency_score": 64.3
  },
  "recommendations": [
    {
      "id": "grow_micro_batch", "category": "memory",
      "parameter": "micro_batch_size", "current": "4", "suggested": "8",
      "impact": "Raise arithmetic intensity per step; typically the cheapest MFU gain available",
      "confidence": 0.72, "auto_applicable": false
    }
  ]
}
```

Recommendations cover memory (batch growth into headroom, activation
checkpointing under pressure), compute (sequence packing, pipeline bubbles,
load imbalance), data (dataloader workers, pre-tokenization), and
communication (gradient sync frequency, stragglers). Each carries
`auto_applicable`: convergence-neutral knobs a training loop may apply
automatically via the Python client's `Tuner`; everything else is advisory,
because an external daemon must not silently change training semantics.
Utilization also ships as metrics (`trainpulse_cluster_efficiency_score`,
`trainpulse_cluster_compute_waste_percent`, `trainpulse_recommendation_active`)
and in the `top` dashboard.

## Python client

`clients/python` ships a zero-dependency package (see its README):

```python
from trainpulse import TrainPulseClient, Tuner

tp = TrainPulseClient()
with tp.step(global_step=step, tokens=batch_tokens):   # times + reports the step
    loss = train_step(batch)

tuner = Tuner(tp)
num_workers = tuner.suggested("dataloader_workers", current=num_workers)
```

## Commands

- `daemon`: collect continuously and expose `/healthz` and `/v1/snapshot`.
- `top`: collect and render a live terminal dashboard.
- `snapshot`: collect once and print JSON.
- `version`: print the build version and commit.

## Reliability semantics

- A failing collector never kills the daemon. Failed collections are logged,
  counted in `collect_errors` (and `trainpulse_collect_errors_total`), and the
  last good snapshot keeps being served with its `last_error` field set.
- In `auto` mode, if `nvidia-smi` fails TrainPulse falls back to the simulator
  and re-probes the real collector every 30 seconds. Snapshots always carry
  `collector` and `simulated` fields — and the `trainpulse_simulated` metric —
  so simulated telemetry can never masquerade as hardware data. Alert on
  `trainpulse_simulated == 1` in production.
- Training samples pushed to `/v1/training` are attached to telemetry frames
  for 30 seconds after receipt (judged by the daemon's clock, so client clock
  skew does not drop samples), then considered stale.

## Security

The API binds to `127.0.0.1` by default. If you expose it beyond localhost,
set a bearer token; every endpoint except `/healthz` and `/v1/version` then
requires `Authorization: Bearer <token>`:

```sh
./trainpulse daemon -addr 0.0.0.0:9876 -auth-token "$TRAINPULSE_TOKEN"
```

or `"auth_token": "..."` in the config file. POST bodies are capped at 1 MiB.

## Observability integrations

TrainPulse does not replace alerting, dashboards, tracing, or experiment tracking. It emits training-aware metrics and diagnostic events that existing tools can consume.

Prometheus/Grafana:

```yaml
scrape_configs:
  - job_name: trainpulse
    static_configs:
      - targets: ["127.0.0.1:9876"]
```

Grafana can use Prometheus as the data source and chart metrics such as:

- `trainpulse_health_score`
- `trainpulse_training_tokens_per_second`
- `trainpulse_training_mfu`
- `trainpulse_gpu_utilization_percent`
- `trainpulse_signal_active`

Datadog:

- Use `/v1/metrics` from a Datadog Agent check, sidecar, or small forwarder.
- Or scrape `/metrics` with the Datadog Agent OpenMetrics integration.
- Use `/v1/events?format=ndjson` as structured diagnostic logs.
- Each metric includes `metric`, `value`, `type`, and optional tags such as model, framework, GPU, signal name, and severity.
- The JSON shape is intentionally simple so it can also feed OpenTelemetry collectors or internal agents.

Other tools:

- Elastic/Splunk/OpenObserve: ingest `/v1/events?format=ndjson`.
- W&B/MLflow/Neptune/Comet/ClearML: log `/v1/metrics` and `/v1/events` as run metrics/artifacts.
- Langfuse/LangSmith/Helicone/Phoenix/Braintrust: attach `/v1/events` as training-run context beside LLM traces and evals.
- NVIDIA DCGM Exporter: keep it for raw GPU telemetry; use TrainPulse for training-aware diagnostics.

See `docs/INTEGRATIONS.md` for the compatibility matrix.

## Configuration

`config.example.json`:

```json
{
  "addr": "127.0.0.1:9876",
  "interval": "1s",
  "mode": "auto",
  "history_size": 120,
  "log_level": "info",
  "log_format": "json",
  "metrics_namespace": "trainpulse"
}
```

CLI flags override config file values.

## User-defined rules

Teams can tune TrainPulse without recompiling it. Add `rules` to the config file:

```json
{
  "rules": [
    {
      "name": "team_low_mfu",
      "field": "training.mfu",
      "operator": "lt",
      "value": 0.35,
      "severity": "warning",
      "score_impact": 12,
      "description": "MFU is below the team-defined efficiency target"
    }
  ]
}
```

Supported operators: `lt`, `lte`, `gt`, `gte`, `eq`, `neq`.

Useful fields:

- `training.tokens_per_sec`
- `training.mfu`
- `training.step_time_ms`
- `training.data_wait_ms`
- `training.tokenizer_wait_ms`
- `training.all_reduce_wait_ms`
- `training.pipeline_bubble_ms`
- `training.checkpoint_ms`
- `gpu.utilization`
- `gpu.memory_used_ratio`
- `gpu.temperature_c`
- `host.load_1`

## Diagnostic knowledge base

Detected signals are turned into root-cause diagnoses by a data-driven
knowledge base, not by compiled-in logic. The built-in knowledge base ships
embedded in the binary (`internal/knowledge/default.json`), so out-of-the-box
behavior is identical whether or not you provide a config file.

Teams can extend or override any diagnosis without recompiling by adding
`diagnoses` to the config file:

```json
{
  "diagnoses": [
    {
      "root_cause": "dataloader_starvation",
      "when_signals": ["dataloader_starvation"],
      "confidence": 0.9,
      "explanation": "GPUs are idling on input; our storage tier is the usual culprit.",
      "actions": ["Increase dataloader workers", "Move the shard to the fast NVMe cache"]
    },
    {
      "root_cause": "team_efficiency_breach",
      "when_signals": ["custom_low_mfu"],
      "confidence": 0.7,
      "explanation": "MFU fell below the team-defined efficiency target set in rules.",
      "actions": ["Check kernel fusion and precision mode", "Open an efficiency ticket"]
    }
  ]
}
```

Semantics:

- `when_signals`: which detected signal names trigger this diagnosis. These are
  the built-in signal names (e.g. `dataloader_starvation`, `low_mfu`,
  `rank_straggler`) or the `name` of any custom rule you defined under `rules`.
- `match`: `any` (default) fires when at least one listed signal is active;
  `all` requires every listed signal to be active.
- An entry whose `root_cause` matches a built-in one **replaces it in place**;
  a new `root_cause` is **appended** after the built-ins. Diagnoses are emitted
  in knowledge-base order.

This pairs with user-defined rules: a custom `rule` produces a signal, and a
custom `diagnosis` turns that signal into an actionable root cause — the full
detection-to-remediation path is configurable.

## LLM coverage

TrainPulse does not need to know only "ML model training." It treats LLM jobs as a richer workload class with model metadata, token throughput, MFU, sequence packing efficiency, distributed rank health, communication stalls, checkpoint IO, and pipeline parallel idle time.

## Architecture

```text
collector
    ↓
stream window
    ↓
anomaly engine
    ↓
correlation engine
    ↓
health scoring
    ↓
root cause inference
    ↓
terminal dashboard / local API
```

## Extension points

TrainPulse now has internal interfaces for community extensions:

- collectors: implement `collector.Collector`
- detectors: implement `anomaly.Detector`
- framework adapters: implement `framework.Adapter`
- plugin bundles: implement `plugin.Plugin`

The first plugin surface is compile-time Go registration. Dynamic binary loading is intentionally deferred until the API and safety model are firmer.

## Linux service

See `packaging/systemd/trainpulse.service`.
