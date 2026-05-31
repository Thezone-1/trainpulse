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

## Commands

- `daemon`: collect continuously and expose `/healthz` and `/v1/snapshot`.
- `top`: collect and render a live terminal dashboard.
- `snapshot`: collect once and print JSON.

## Observability integrations

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
- Each metric includes `metric`, `value`, `type`, and optional tags such as model, framework, GPU, signal name, and severity.
- The JSON shape is intentionally simple so it can also feed OpenTelemetry collectors or internal agents.

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
