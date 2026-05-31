# TrainPulse

TrainPulse is a lightweight Go daemon for predictive diagnostics of AI training systems. It collects runtime telemetry, runs low-latency anomaly checks, scores training health, infers likely root causes, and exposes real-time terminal-native stats.

## First slice

- Linux daemon shape with local HTTP snapshot API.
- NVIDIA GPU telemetry through `nvidia-smi`.
- Host memory and load telemetry from `/proc`.
- Simulation mode for development without GPUs.
- Real-time health scoring and terminal dashboard.
- Early diagnostic rules for dataloader starvation, GPU underutilization, sync imbalance, memory pressure, thermal instability, and throughput collapse.

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

Fetch one JSON snapshot:

```sh
curl http://127.0.0.1:9876/v1/snapshot
```

## Commands

- `daemon`: collect continuously and expose `/healthz` and `/v1/snapshot`.
- `top`: collect and render a live terminal dashboard.
- `snapshot`: collect once and print JSON.

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

## Linux service

See `packaging/systemd/trainpulse.service`.
